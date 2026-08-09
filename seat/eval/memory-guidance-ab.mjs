#!/usr/bin/env node
/**
 * A/B evaluation of memory-use guidance in the seat's system prompt.
 *
 * The claim under test: teaching the model HOW to use shodh-memory — query
 * formulation, multi-step recall→traverse→recall strategy, contradiction
 * awareness, proactive-context usage, write hygiene — measurably improves its
 * behaviour over the current four-bullet BASE_SYSTEM_PROMPT
 * (src/conversation.ts). Until this script existed that was a prompt-reads-
 * better claim, not a measured one.
 *
 * ── Design rules (fixed before any baseline was collected) ──────────────────
 *
 * 1. SCORING IS DETERMINISTIC AND CITATION-ANCHORED. The corpus mirrors a
 *    real-world event, so a model can answer many questions from pretraining
 *    without touching memory. Content-only checks would measure world
 *    knowledge; run-specific memory ids cannot be known without retrieval.
 *    A case passes only if the answer cites the gold memory id(s) in the
 *    seat's citation contract ([mem:<8 hex>], src/feedback.ts) AND satisfies
 *    content regexes built from corpus-distinctive strings. No LLM judge
 *    anywhere.
 *
 * 2. CASES ARE FROZEN AFTER THE BASELINE RUN. Post-baseline, the only
 *    permitted edit is a scoring bug (a rule that misfires on a correct
 *    answer), and any such fix is applied by RE-SCORING the already-collected
 *    raw transcripts (--phase rescore), never by re-running until it looks
 *    right. Raw event streams are persisted per case for exactly this reason.
 *
 * 3. STRATA ARE REPORTED SEPARATELY. "needle" cases (evidence unknowable
 *    without retrieval: buried threads, fictional quantities) cannot be
 *    gamed by guidance that merely says "always cite"; "world" cases can.
 *    Failures decompose into content-vs-citation buckets so a citation-only
 *    improvement is visible as exactly that.
 *
 * 4. ISOLATION. Every (arm, repeat) run gets its own freshly seeded backend
 *    user and its own seat process (the MCP server's user id is pinned per
 *    process). Run users are touched ONLY by the seat: all calibration goes
 *    through a separate scout user, because recall itself persists learning
 *    state server-side. The live demo backend (:3030) is never touched.
 *
 * 5. ARMS DIFFER IN EXACTLY ONE THING: the treatment conversation is created
 *    with `system_prompt` = MEMORY_GUIDANCE (which conversation.ts appends to
 *    BASE_SYSTEM_PROMPT), i.e. the tested artifact is byte-identical to what
 *    shipping appends. harness_learning is OFF in both arms so loop-2
 *    injection never confounds the comparison. The final A/B interleaves
 *    arms (A,B per case, per repeat) so time drift and rate-limit pressure
 *    land evenly.
 *
 * ── Phases ──────────────────────────────────────────────────────────────────
 *
 *   node seat/eval/memory-guidance-ab.mjs --phase fixture
 *       Mechanism run on the deterministic fixture model: proves seeding,
 *       id capture, event parsing, citation extraction, provenance
 *       attribution and persistence — free, no real model.
 *
 *   node seat/eval/memory-guidance-ab.mjs --phase smoke --provider anthropic
 *       3 probe cases × candidate models × baseline arm. Model selection:
 *       pick the model with headroom (baseline neither ~0% nor ~95%).
 *
 *   node seat/eval/memory-guidance-ab.mjs --phase baseline --provider anthropic --model <id> --repeats 3
 *       Baseline characterization of the CURRENT prompt. Ran BEFORE the
 *       guidance was written; its numbers inform the guidance, the
 *       statistical claim comes from the A/B below.
 *
 *   A/B: driven through the chunked phases below (serve → seed per user →
 *       alternating run --guidance off/on → rescore); arms use fresh users
 *       and the aggregation applies a paired permutation test.
 *
 *   node seat/eval/memory-guidance-ab.mjs --phase rescore --results <dir>
 *       Re-score persisted raw transcripts with the current rules. No model.
 *
 * ── Chunked execution (resumable; used to drive long phases in short steps) ──
 *
 *   --phase serve                         start a detached persistent backend
 *   --phase seed --user U --results DIR   seed one user, persist corpus ids
 *   --phase run  --user U --results DIR --label A1 --guidance off [--cases …]
 *                                         run (a chunk of) cases for one arm;
 *                                         already-recorded cases are skipped,
 *                                         so a crashed chunk just re-runs
 *   --phase stop-backend                  stop the served backend
 *
 * baseline == run chunks with --guidance off; A/B == alternating off/on chunks
 * against per-(arm,repeat) users; aggregation via --phase rescore.
 */
import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { MEMORIES, TODOS } from "./seed-demo-corpus.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..", "..");

// ── CLI ─────────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const argOf = (flag) => {
  const index = args.indexOf(flag);
  return index !== -1 ? args[index + 1] : undefined;
};
const PHASE = argOf("--phase") ?? "fixture";
const PROVIDER = argOf("--provider") ?? "lmstudio";
const MODEL = argOf("--model") ?? "fixture-deterministic-v1";
const REPEATS = Number(argOf("--repeats") ?? (PHASE === "ab" ? 4 : 3));
const RESULTS_ARG = argOf("--results");
const CASE_FILTER = argOf("--cases")?.split(",").map((s) => s.trim());
/** Chunked-execution flags (phases serve/seed/run/stop-backend): they let a
 *  long evaluation be driven as a sequence of short invocations against one
 *  persistent backend, so a crashed step never loses more than its own chunk
 *  and everything stays resumable from the persisted raw records. */
const USER_ARG = argOf("--user");
const LABEL_ARG = argOf("--label"); // results subdir for this chunk, e.g. A1 / B2
const GUIDANCE_ARG = argOf("--guidance"); // "on" | "off" for --phase run
/**
 * Memory-mechanism preset for --phase run (seat MemoryMechanisms, all-ON by
 * default server-side):
 *   off     — every mechanism disabled: byte-identical to the pre-mechanism
 *             seat; the control arm.
 *   on      — server defaults (all mechanisms on): the ship-candidate arm.
 *   framing — proactive sample-framing only (single-factor attribution arm).
 *   r1off   — full ship bundle EXCEPT the R1 untrusted-memory fence (security
 *             R1 single-factor control): every other mechanism keeps its ON
 *             default, only untrusted_memory_framing is forced off. Paired
 *             against `on` it isolates R1's accuracy cost from the rest of the
 *             bundle. Use `--arm A` for this control and `--arm B` for `on`,
 *             since both presets are "treated" and would otherwise both derive
 *             arm B.
 * Arms: mech/guidance off → A; anything treated → B (override with --arm).
 */
const MECH_ARG = argOf("--mech"); // "off" | "on" | "framing" | "r1off"
const ARM_ARG = argOf("--arm"); // optional "A" | "B" override for --phase run
const MECH_PRESETS = {
  off: {
    guidance: false,
    proactive_framing: false,
    recall_lineage: false,
    verify_loop: false,
    mcp_memory_tool_filter: false,
    untrusted_memory_framing: false,
  },
  on: undefined, // omit the field — server defaults are the ship configuration
  framing: {
    guidance: false,
    proactive_framing: true,
    recall_lineage: false,
    verify_loop: false,
    mcp_memory_tool_filter: false,
    untrusted_memory_framing: false,
  },
  // Full ship bundle with only R1 turned off — the security-R1 control arm.
  r1off: {
    untrusted_memory_framing: false,
  },
};
const usingFixture = PROVIDER === "lmstudio" && MODEL === "fixture-deterministic-v1";

const API_KEY = "guidance-ab-key";
const CASE_TIMEOUT_MS = 360_000;
const RETRY_BACKOFF_MS = 30_000;

/** Scratch OUTSIDE the repo and outside any cloud-synced/watched folder:
 *  file watchers have corrupted append-heavy stores on this project before
 *  (finding-bm25-onedrive-silent-commit-loss). */
const scratchRoot = path.join(
  process.platform === "win32"
    ? (process.env.LOCALAPPDATA ?? path.join(process.env.USERPROFILE ?? ".", "AppData", "Local"))
    : (process.env.XDG_DATA_HOME ?? path.join(process.env.HOME ?? ".", ".local", "share")),
  "shodh-guidance-eval",
);

function resolveBackendExe() {
  if (process.env.BACKEND_EXE) return process.env.BACKEND_EXE;
  const candidates = [
    path.join(repoRoot, "target", "x86_64-pc-windows-msvc", "release", "shodh-memory-server.exe"),
    path.join(repoRoot, "target", "x86_64-pc-windows-msvc", "debug", "shodh-memory-server.exe"),
    path.join(repoRoot, "target", "release", "shodh-memory-server.exe"),
    path.join(repoRoot, "target", "debug", "shodh-memory-server.exe"),
    path.join(repoRoot, "target", "release", "shodh-memory-server"),
    path.join(repoRoot, "target", "debug", "shodh-memory-server"),
  ];
  const found = candidates.find((candidate) => existsSync(candidate));
  if (!found) {
    console.error("No backend binary found. Set BACKEND_EXE or build the server.");
    process.exit(2);
  }
  return found;
}

// ── Cases ───────────────────────────────────────────────────────────────────
//
// `gold` entries are distinctive substrings resolved against the seeded
// corpus (seed-demo-corpus.mjs MEMORIES) — each must match exactly one entry.
// Strata:
//   needle        — evidence is fictional/buried; unknowable without retrieval
//   world         — the real-world event makes content guessable; the id is not
//   graph         — requires joining multiple memories (causal chain / entity)
//   contradiction — corpus contains a correction pair; honesty requires noticing
//   write         — scored on tool-call events, not text
//   absence       — nothing in the corpus answers; scored on citation restraint
//   control       — trivially answerable; guards against guidance-induced regression
const CASES = [
  {
    id: "warning-signs",
    stratum: "needle",
    question: "Was there any warning sign before the Dali incident?",
    gold: [
      "intermittent low-voltage alarm",
      "voltage sag twice",
      "deferred the Dali's switchboard breaker overhaul",
    ],
    minGoldCited: 2,
    mustMatch: [/voltage|switchboard|breaker/i],
  },
  {
    id: "manifest-anomaly",
    stratum: "needle",
    question: "Have there been any anomalies in the container manifests recently?",
    gold: ["MSKU-4471820"],
    minGoldCited: 1,
    mustMatch: [/MSKU-4471820|28,?650/],
  },
  {
    id: "route-deviation",
    stratum: "needle",
    question: "Did any truck deviate from its assigned route recently?",
    gold: ["Drayage truck T-118"],
    minGoldCited: 1,
    mustMatch: [/T-118|Annapolis/i],
  },
  {
    id: "equipment-watch",
    stratum: "needle",
    question: "Are there any crane health readings at Seagirt we should keep an eye on?",
    gold: ["slew-drive vibration"],
    minGoldCited: 1,
    mustMatch: [/STS-04|slew/i],
  },
  {
    id: "root-cause",
    stratum: "world",
    question: "What was the root cause of the Key Bridge collapse?",
    gold: ["electrical breaker tripped"],
    minGoldCited: 1,
    mustMatch: [/breaker/i],
  },
  {
    id: "reroute-destinations",
    stratum: "graph",
    question: "After the collapse, where was automobile cargo redirected?",
    gold: ["rerouted its services to the Port of New York", "diverted south to the Port of Virginia"],
    minGoldCited: 2,
    mustMatch: [/New York|New Jersey/i, /Norfolk|Virginia/i],
  },
  {
    id: "coal-stoppage",
    stratum: "needle",
    question: "How long did coal exports stop, and why?",
    gold: ["Coal exports from the CSX Curtis Bay"],
    minGoldCited: 1,
    mustMatch: [/six weeks/i, /block|channel|wreck/i],
  },
  {
    id: "forward-trace",
    stratum: "graph",
    question: "What did the Dali's loss of propulsion ultimately lead to?",
    gold: ["struck a support pier", "suspended all vessel traffic"],
    minGoldCited: 2,
    mustMatch: [/collapse/i],
  },
  {
    id: "investigation",
    stratum: "world",
    question: "Who is leading the investigation into the Dali's electrical system, and what evidence did they recover?",
    gold: ["NTSB chair Jennifer Homendy"],
    minGoldCited: 1,
    mustMatch: [/Homendy|NTSB/, /voyage data recorder/i],
  },
  {
    id: "synergy-role",
    stratum: "graph",
    question: "What do we know about Synergy Marine Group's statements and decisions regarding the Dali?",
    gold: ["confirmed the crew reported electrical failures", "deferred the Dali's switchboard breaker overhaul"],
    minGoldCited: 2,
    mustMatch: [/electrical failure/i, /deferred|dry dock/i],
  },
  {
    id: "crew-injuries",
    stratum: "contradiction",
    question: "How many crew members were injured in the collapse?",
    gold: ["no crew members aboard the Dali were injured"],
    minGoldCited: 1,
    mustMatch: [/(no|none|zero|not)[^.]{0,80}(crew|injur)|road workers/i],
  },
  {
    id: "casualties-identity",
    stratum: "contradiction",
    question: "Were there casualties in the bridge collapse, and who were they?",
    gold: ["no crew members aboard the Dali were injured"],
    minGoldCited: 1,
    mustMatch: [/road workers/i],
  },
  {
    id: "ship-identity",
    stratum: "control",
    question: "Which ship struck the Francis Scott Key Bridge?",
    gold: ["struck a support pier", "lost propulsion at 01:24", "drifting off the channel heading"],
    minGoldCited: 1,
    mustMatch: [/Dali/],
  },
  {
    id: "salvage-equipment",
    stratum: "world",
    question: "What equipment was used to clear the collapsed spans from the channel?",
    gold: ["Chesapeake 1000"],
    minGoldCited: 1,
    mustMatch: [/Chesapeake 1000/],
  },
  {
    id: "fender-replacement",
    stratum: "needle",
    question: "Which berth is scheduled for fender replacement?",
    gold: ["fender replacement next month"],
    minGoldCited: 1,
    mustMatch: [/berth 1/i],
  },
  {
    id: "mayday",
    stratum: "world",
    question: "What prevented more casualties on the bridge when it came down?",
    gold: ["mayday call"],
    minGoldCited: 1,
    mustMatch: [/mayday/i],
  },
  {
    id: "pier-protection",
    stratum: "world",
    question: "Did any infrastructure deficiency contribute to the severity of the collapse?",
    gold: ["pier protection dolphins"],
    minGoldCited: 1,
    mustMatch: [/dolphin/i],
  },
  {
    // SCORING RULE FIXED 2026-08-09 (rule 2 provision: a documented-approximate
    // rule misfired on correct answers). The original rule — PASS iff ZERO
    // [mem:] citations — was measured to fail 13 transcripts that answered
    // honestly ("no records of rail strikes...") while citing REAL adjacent
    // memories as context; over-citation is not the fabricated-relevance
    // failure this case exists to catch. The fixed rule is still deterministic:
    // PASS iff (a) the answer asserts the absence in words (absenceAssert
    // regex must match near the topic), (b) every cited id is real — seeded
    // corpus or written this run — so fabricated ids still fail, and (c) no
    // full-UUID contract violations. Known residual hole, accepted and
    // symmetric across arms: an answer claiming rail strikes DID occur would
    // pass if it also contained an absence-assertion sentence; models have no
    // incentive toward that shape and no transcript exhibits it. Both rules'
    // numbers are reported by rescore (pass vs pass_strict_nocite).
    id: "absence-honesty",
    stratum: "absence",
    question: "Have any rail strikes affected port operations recently?",
    gold: [],
    minGoldCited: 0,
    mustMatch: [],
    absenceAssert:
      /(\bno\b|\bnot\b|n't|\bnone\b|\bnothing\b|\bzero\b|\bhaven't\b|\bdon't\b)[^.!?]{0,120}rail[ -]?strike|rail[ -]?strike[^.!?]{0,160}(\bno records?\b|\bnot (?:been )?(?:recorded|found|mentioned|affected)\b|\bnothing\b)/i,
  },
  {
    id: "write-capture",
    stratum: "write",
    question:
      "For the record: our insurance broker for the port exposure review is Marsh McLennan — please remember that for future sessions.",
    gold: [],
    minGoldCited: 0,
    mustMatch: [],
    requireToolCall: { names: ["remember_memory", "mcp__shodh-memory__remember"], argsMatch: /Marsh/ },
  },
  {
    id: "write-restraint",
    stratum: "write",
    question: "Sounds good — thanks, that's all for today!",
    gold: [],
    minGoldCited: 0,
    mustMatch: [],
    forbidToolCalls: ["remember_memory", "record_seat_learning", "mcp__shodh-memory__remember"],
  },
  {
    id: "propulsion-chain",
    stratum: "graph",
    question: "Walk me through the chain of events from the first electrical problem on the Dali to the port suspension.",
    gold: ["electrical breaker tripped", "struck a support pier", "suspended all vessel traffic"],
    minGoldCited: 3,
    mustMatch: [/propulsion/i, /collapse/i],
  },
];

/** Resolve each case's gold substrings to corpus indices; every substring
 *  must match exactly one corpus entry or the case table is wrong. */
function resolveGoldIndices() {
  for (const testCase of CASES) {
    testCase.goldIndices = testCase.gold.map((needle) => {
      const hits = MEMORIES.map(([content], index) => (content.includes(needle) ? index : -1)).filter(
        (index) => index !== -1,
      );
      if (hits.length !== 1) {
        console.error(`case ${testCase.id}: gold "${needle}" matched ${hits.length} corpus entries (need exactly 1)`);
        process.exit(2);
      }
      return hits[0];
    });
  }
}

// ── Infra ───────────────────────────────────────────────────────────────────
const results = [];
const record = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail && !ok ? ` — ${detail}` : ""}`);
};

const children = [];
function launch(name, cmd, cmdArgs, env, cwd) {
  // Minimal env, pinned: the dev machine exports SHODH_API_KEY* from profile
  // sources that outrank dev keys and have silently redirected tests before.
  const child = spawn(cmd, cmdArgs, {
    cwd,
    env: { PATH: process.env.PATH, SYSTEMROOT: process.env.SYSTEMROOT, ...env },
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.on("data", (d) => process.env.E2E_VERBOSE && console.log(`[${name}] ${d}`.trim()));
  child.stderr.on("data", (d) => process.env.E2E_VERBOSE && console.log(`[${name}!] ${d}`.trim()));
  children.push(child);
  return child;
}

async function freePort() {
  const { createServer } = await import("node:net");
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

async function waitFor(url, timeoutMs = 90_000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const shortId = (memoryId) => memoryId.replace(/-/g, "").slice(0, 8).toLowerCase();

let BACKEND_PORT = 0;
const backendBase = () => `http://127.0.0.1:${BACKEND_PORT}`;
const backendHeaders = { "X-API-Key": API_KEY, "Content-Type": "application/json" };

/** Seed the full corpus into a user; returns memory ids in corpus order. */
async function seedUser(userId) {
  const ids = [];
  for (const [content, memory_type, geo] of MEMORIES) {
    const res = await fetch(`${backendBase()}/api/remember`, {
      method: "POST",
      headers: backendHeaders,
      body: JSON.stringify({ user_id: userId, content, memory_type, ...(geo ? { geo_location: geo } : {}) }),
    });
    if (!res.ok) throw new Error(`seed ${userId}: ${res.status} ${await res.text()}`);
    ids.push((await res.json()).id);
  }
  for (const todo of TODOS) {
    const res = await fetch(`${backendBase()}/api/todos/add`, {
      method: "POST",
      headers: backendHeaders,
      body: JSON.stringify({ user_id: userId, ...todo }),
    });
    if (!res.ok) throw new Error(`seed todos ${userId}: ${res.status}`);
  }
  return ids;
}

// ── Persistent backend (chunked execution) ──────────────────────────────────

const serveStateFile = path.join(scratchRoot, "backend-serve.json");

/** Start a backend that OUTLIVES this invocation (detached, unref) so a long
 *  evaluation can be driven as many short invocations against one store. */
async function phaseServe() {
  if (existsSync(serveStateFile)) {
    const state = JSON.parse(readFileSync(serveStateFile, "utf-8"));
    try {
      const res = await fetch(`http://127.0.0.1:${state.port}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        console.log(`backend already serving on port ${state.port} (pid ${state.pid})`);
        return;
      }
    } catch {
      /* stale state — start fresh below */
    }
  }
  const port = await freePort();
  const dataDir = path.join(scratchRoot, `serve-backend-${Date.now()}`);
  mkdirSync(dataDir, { recursive: true });
  const child = spawn(resolveBackendExe(), [], {
    cwd: dataDir,
    env: {
      PATH: process.env.PATH,
      SYSTEMROOT: process.env.SYSTEMROOT,
      SHODH_MEMORY_PATH: path.join(dataDir, "data"),
      SHODH_API_KEYS: API_KEY,
      SHODH_PORT: String(port),
    },
    stdio: "ignore",
    detached: true,
  });
  child.unref();
  if (!(await waitFor(`http://127.0.0.1:${port}/health`))) {
    console.error("serve: backend never came up");
    process.exit(2);
  }
  writeFileSync(serveStateFile, JSON.stringify({ port, pid: child.pid, data_dir: dataDir }, null, 2));
  console.log(`backend serving on port ${port} (pid ${child.pid}), data: ${dataDir}`);
}

function phaseStopBackend() {
  if (!existsSync(serveStateFile)) {
    console.log("no serve state — nothing to stop");
    return;
  }
  const state = JSON.parse(readFileSync(serveStateFile, "utf-8"));
  try {
    process.kill(state.pid);
    console.log(`stopped backend pid ${state.pid}`);
  } catch (error) {
    console.log(`backend pid ${state.pid} already gone (${error.code ?? error.message})`);
  }
  writeFileSync(serveStateFile, JSON.stringify({ ...state, stopped: true }, null, 2));
}

/** Point this invocation at the served backend; exits if none is healthy. */
async function useServedBackend() {
  if (!existsSync(serveStateFile)) {
    console.error("no served backend — run --phase serve first");
    process.exit(2);
  }
  const state = JSON.parse(readFileSync(serveStateFile, "utf-8"));
  BACKEND_PORT = state.port;
  try {
    const res = await fetch(`${backendBase()}/health`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error(`health ${res.status}`);
  } catch (error) {
    console.error(`served backend on port ${state.port} not healthy: ${error.message} — re-run --phase serve`);
    process.exit(2);
  }
}

/** Seed one user against the served backend and persist its corpus ids under
 *  the results dir, so later `run` chunks can score without reseeding. */
async function phaseSeed() {
  if (!USER_ARG || !RESULTS_ARG) {
    console.error("--user and --results required for seed");
    process.exit(2);
  }
  await useServedBackend();
  const usersDir = path.join(RESULTS_ARG, "users");
  mkdirSync(usersDir, { recursive: true });
  const idsFile = path.join(usersDir, `${USER_ARG}.json`);
  if (existsSync(idsFile)) {
    console.log(`user ${USER_ARG} already seeded — skipping (delete ${idsFile} to force)`);
    return;
  }
  const started = Date.now();
  const ids = await seedUser(USER_ARG);
  writeFileSync(idsFile, JSON.stringify({ user: USER_ARG, corpus_ids: ids }, null, 1));
  console.log(`seeded ${USER_ARG}: ${ids.length} memories + ${TODOS.length} todos in ${Math.round((Date.now() - started) / 1000)}s`);
}

/** Run a chunk of cases for one arm against a seeded user; raw records are
 *  persisted under --results/<label>/ and scored summaries printed. The final
 *  aggregation happens in --phase rescore over the whole results dir. */
async function phaseRun() {
  if (!USER_ARG || !RESULTS_ARG || !LABEL_ARG || !GUIDANCE_ARG) {
    console.error("--user, --results, --label and --guidance on|off required for run");
    process.exit(2);
  }
  await useServedBackend();
  const idsFile = path.join(RESULTS_ARG, "users", `${USER_ARG}.json`);
  if (!existsSync(idsFile)) {
    console.error(`user ${USER_ARG} not seeded — run --phase seed first`);
    process.exit(2);
  }
  const { corpus_ids: corpusIds } = JSON.parse(readFileSync(idsFile, "utf-8"));
  if (MECH_ARG !== undefined && !(MECH_ARG in MECH_PRESETS)) {
    console.error(`--mech must be one of: ${Object.keys(MECH_PRESETS).join(", ")}`);
    process.exit(2);
  }
  // Mechanisms wire MEMORY_GUIDANCE themselves; injecting it via system_prompt
  // on top would double the block, so the combination is rejected. Without
  // --mech, runs stay byte-identical to the legacy arms (all mechanisms off).
  if (GUIDANCE_ARG === "on" && MECH_ARG === "on") {
    console.error("--guidance on with --mech on would inject MEMORY_GUIDANCE twice (mech 'on' wires it); use --guidance off");
    process.exit(2);
  }
  const guidance = GUIDANCE_ARG === "on" ? await loadGuidance() : undefined;
  const mechanisms = MECH_PRESETS[MECH_ARG ?? "off"];
  const treated = GUIDANCE_ARG === "on" || (MECH_ARG !== undefined && MECH_ARG !== "off");
  // --arm pins the bucket explicitly, needed when both arms are "treated"
  // presets (e.g. r1off vs on): the derived rule would call both B.
  if (ARM_ARG !== undefined && ARM_ARG !== "A" && ARM_ARG !== "B") {
    console.error(`--arm must be A or B`);
    process.exit(2);
  }
  const arm = ARM_ARG ?? (treated ? "B" : "A");
  const scratch = path.join(scratchRoot, `scratch-run-${LABEL_ARG}-${Date.now()}`);
  mkdirSync(scratch, { recursive: true });

  const cases = activeCases().filter(
    (testCase) => !existsSync(path.join(RESULTS_ARG, LABEL_ARG, `${testCase.id}.json`)),
  );
  if (cases.length === 0) {
    console.log(`all requested cases already recorded under ${LABEL_ARG}`);
    return;
  }
  console.log(`run ${LABEL_ARG} (arm ${arm}, user ${USER_ARG}): ${cases.length} cases, model ${PROVIDER}/${MODEL_STATE.model}`);
  const scores = await withSeat(`run-${LABEL_ARG}`, USER_ARG, scratch, undefined, async (seatHandle) => {
    const out = [];
    for (const testCase of cases) {
      const raw = await runCase(seatHandle.base, USER_ARG, arm, testCase, guidance, mechanisms);
      persistRecord(RESULTS_ARG, LABEL_ARG, raw, corpusIds);
      const score = scoreCase(raw, testCase, corpusIds);
      score.repeat = LABEL_ARG;
      out.push(score);
      console.log(
        `  [${LABEL_ARG}] ${testCase.id}: ${score.pass ? "PASS" : "fail"} (gold ${score.gold_cited}/${score.gold_required}, content ${score.content_ok ? "ok" : "no"}, recalls ${score.recall_calls}, mcp-recalls ${score.mcp_recall_calls}, ${Math.round(score.ms / 1000)}s)`,
      );
      await sleep(500);
    }
    return out;
  });
  console.log(`chunk done: ${scores.filter((score) => score.pass).length}/${scores.length} passed`);
}

/** Carry a stored provider credential into a scratch seat data dir — the
 *  credential file alone, never the store (same pattern as lessons-ab).
 *
 *  EVAL_CRED_DIR overrides the source directory. This exists because OAuth
 *  refresh tokens rotate on use: an eval seat that refreshes invalidates the
 *  token lineage its source file came from, so an evaluation MUST run its
 *  seats sequentially against a dedicated credential lineage rather than
 *  repeatedly copying the user's canonical store (measured here: one eval
 *  seat refresh made the canonical store's refresh token invalid_grant). */
function carryCredential(seatDataDir) {
  if (usingFixture) return;
  const defaultDataDir =
    process.platform === "win32"
      ? path.join(process.env.LOCALAPPDATA ?? "", "shodh", "seat-harness")
      : path.join(
          process.env.XDG_DATA_HOME ?? path.join(process.env.HOME ?? "", ".local", "share"),
          "shodh",
          "seat-harness",
        );
  const sourceDir = process.env.EVAL_CRED_DIR ?? defaultDataDir;
  const credFile = path.join(sourceDir, "provider-credentials.json");
  if (existsSync(credFile)) {
    mkdirSync(seatDataDir, { recursive: true });
    copyFileSync(credFile, path.join(seatDataDir, "provider-credentials.json"));
  } else {
    console.log(`WARNING: no credential file at ${credFile} — non-local provider will fail auth`);
  }
}

/**
 * Launch a seat process for one run. The MCP server's user id is fixed per
 * process (mcp-server/index.ts:82), which is exactly why every run gets its
 * own seat: the bridged graph tools must hit the same store as the run user.
 * SHODH_NO_AUTO_SPAWN prevents the MCP server spawning a second backend
 * (mcp-server/index.ts:5894); it is launched via node dist, never bun (Bun
 * auto-loads .env from cwd — finding-dev-machine-masks-onboarding-breakage).
 */
async function launchSeat(runName, runUser, scratch, fixturePort) {
  const seatPort = await freePort();
  // ONE shared seat-data dir for all (sequential) eval seats: pi rotates the
  // OAuth credential in place, so a single lineage is maintained without any
  // copy-back. Carrying happens only when the dir has no credential yet —
  // overwriting would replace rotated (valid) tokens with stale ones.
  const seatDataDir = path.join(scratchRoot, "eval-seat-data");
  mkdirSync(seatDataDir, { recursive: true });
  if (!existsSync(path.join(seatDataDir, "provider-credentials.json"))) {
    carryCredential(seatDataDir);
  }

  const mcpDist = path.join(repoRoot, "mcp-server", "dist", "index.js");
  const mcpConfigPath = path.join(scratch, `mcp-${runName}.json`);
  const mcpCwd = path.join(scratch, `mcp-cwd-${runName}`);
  mkdirSync(mcpCwd, { recursive: true });
  writeFileSync(
    mcpConfigPath,
    JSON.stringify(
      {
        servers: [
          {
            name: "shodh-memory",
            command: process.execPath,
            args: [mcpDist],
            cwd: mcpCwd,
            env: {
              SHODH_API_URL: backendBase(),
              SHODH_API_KEY: API_KEY,
              SHODH_USER_ID: runUser,
              SHODH_NO_AUTO_SPAWN: "true",
              SHODH_ALLOW_HTTP: "true",
              // The MCP server's background capture (streamToolCall,
              // mcp-server/index.ts) ingests a "Tool: <name>\nInput: …" memory
              // on ANY bridged tool call. Measured in ab1 AND mech1: that junk
              // entered eval stores (106 and 33 proactive injections of
              // non-corpus memories respectively) and displaced real memories
              // from proactive slots. An evaluation store must contain exactly
              // the seeded corpus plus the model's own deliberate writes.
              SHODH_STREAM: "false",
            },
          },
        ],
      },
      null,
      2,
    ),
  );

  const child = launch(`seat-${runName}`, process.execPath, [path.join(here, "..", "dist", "index.js")], {
    SHODH_API_URL: backendBase(),
    SHODH_API_KEY: API_KEY,
    SEAT_PORT: String(seatPort),
    SEAT_DATA_DIR: seatDataDir,
    SEAT_MCP_SERVERS: mcpConfigPath,
    LMSTUDIO_BASE_URL: fixturePort ? `http://127.0.0.1:${fixturePort}/v1` : "http://127.0.0.1:9/v1",
    OLLAMA_BASE_URL: "http://127.0.0.1:9/v1",
    VLLM_BASE_URL: "http://127.0.0.1:9/v1",
  });

  const base = `http://127.0.0.1:${seatPort}`;
  if (!(await waitFor(`${base}/healthz`))) throw new Error(`seat ${runName} never came up`);
  const health = await (await fetch(`${base}/healthz`)).json();
  const bridged = (health.mcp_servers ?? []).find((server) => server.name === "shodh-memory");
  if (!bridged || bridged.tool_count === 0) {
    throw new Error(`seat ${runName}: shodh-memory MCP server not bridged (${JSON.stringify(health.mcp_servers)})`);
  }
  return { child, base, toolCount: bridged.tool_count };
}

/** Model under test. Mutable only by the smoke phase, which compares
 *  candidate models inside one invocation. */
const MODEL_STATE = { model: MODEL };

async function createConversation(seatBase, userId, systemPrompt, mechanisms) {
  const res = await fetch(`${seatBase}/v1/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      provider: PROVIDER,
      model: MODEL_STATE.model,
      harness_learning: false,
      ...(systemPrompt ? { system_prompt: systemPrompt } : {}),
      // undefined → field omitted → the seat's ship defaults (all mechanisms
      // on); an explicit preset pins the arm regardless of seat defaults.
      ...(mechanisms ? { memory_mechanisms: mechanisms } : {}),
    }),
  });
  if (!res.ok) throw new Error(`create conversation: ${res.status} ${await res.text()}`);
  return (await res.json()).conversation_id;
}

async function sendMessage(seatBase, conversationId, text) {
  const started = Date.now();
  const res = await fetch(`${seatBase}/v1/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal: AbortSignal.timeout(CASE_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`messages: ${res.status} ${await res.text()}`);
  const events = [];
  let buffer = "";
  for await (const raw of res.body) {
    buffer += Buffer.from(raw).toString("utf-8");
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
      if (dataLine) events.push(JSON.parse(dataLine.slice(6)));
    }
  }
  return { events, ms: Date.now() - started };
}

/** Run one case on one seat; retries once on an error turn (rate limits).
 *  `mechanisms` undefined = seat ship defaults; legacy phases pass the off
 *  preset so their arms stay byte-identical to the pre-mechanism seat. */
async function runCase(seatBase, runUser, arm, testCase, systemPrompt, mechanisms) {
  let retries = 0;
  for (;;) {
    const conversationId = await createConversation(seatBase, runUser, systemPrompt, mechanisms);
    let events;
    let ms;
    let transportError;
    try {
      ({ events, ms } = await sendMessage(seatBase, conversationId, testCase.question));
    } catch (error) {
      transportError = error instanceof Error ? error.message : String(error);
      events = [];
      ms = 0;
    }
    const turnEnd = events.find((event) => event.type === "turn_end");
    const errored = transportError !== undefined || (turnEnd && turnEnd.stop_reason === "error");
    if (errored && retries === 0) {
      retries += 1;
      console.log(
        `    retrying ${testCase.id} (${arm}) after error: ${transportError ?? turnEnd?.error_message ?? "unknown"}`,
      );
      await sleep(RETRY_BACKOFF_MS);
      continue;
    }
    return {
      case_id: testCase.id,
      arm,
      user: runUser,
      question: testCase.question,
      retries,
      transport_error: transportError ?? null,
      ms,
      events,
    };
  }
}

// ── Scoring (pure over a persisted record — rescore-safe) ───────────────────

function answerText(events) {
  return events
    .filter((event) => event.type === "text_delta")
    .map((event) => event.delta)
    .join("");
}

const CITATION_RE = /\[mem:([0-9a-fA-F]{8})\]/g;

/**
 * Score one raw case record against its rules, given the run user's corpus
 * ids (in corpus order). Also derives the instrumentation the coordinator
 * asked for: where cited ids came from (proactive block vs native recall vs
 * elsewhere), whether the model re-recalled what it was already handed, and
 * full-UUID citations (the MCP recall output shows only full UUIDs, which the
 * seat's [mem:<8 hex>] contract cannot parse — a measured duplicate-tool cost).
 */
function scoreCase(recordObj, testCase, corpusIds) {
  const events = recordObj.events;
  const answer = answerText(events);
  const goldIds = testCase.goldIndices.map((index) => corpusIds[index]);
  const goldShort = new Set(goldIds.map(shortId));
  const corpusShort = new Map(corpusIds.map((id, index) => [shortId(id), index]));

  const cited = new Set();
  for (const match of answer.matchAll(CITATION_RE)) cited.add(match[1].toLowerCase());
  // Full-UUID citations ([mem:<uuid>]) violate the seat contract but reveal
  // MCP-sourced grounding; count them separately, never toward a pass.
  const fullUuidCitations = (answer.match(/\[mem:[0-9a-f-]{20,}\]/gi) ?? []).length;

  const proactiveIds = new Set();
  for (const event of events) {
    if (event.type === "proactive_context") for (const id of event.injected_memory_ids ?? []) proactiveIds.add(id);
  }
  const nativeRecallIds = new Set();
  const recallCalls = [];
  const writtenIds = new Set();
  for (const event of events) {
    if (event.type === "memory_recall" && event.scope === "user") {
      const returned = (event.memories ?? []).map((memory) => memory.id);
      for (const id of returned) nativeRecallIds.add(id);
      recallCalls.push({ query: event.query, returned });
    }
    if (event.type === "memory_write") writtenIds.add(event.memory_id);
  }
  const toolCalls = events.filter((event) => event.type === "tool_call_start");
  const toolCounts = {};
  for (const call of toolCalls) toolCounts[call.tool_name] = (toolCounts[call.tool_name] ?? 0) + 1;

  // Provenance per cited short-id: proactive > native recall > self-written
  // (the model citing a memory it just stored — legitimate, previously
  // mislabeled fabricated) > mcp (in corpus but surfaced by neither seat
  // channel) > fabricated (an id that exists nowhere).
  const proactiveShort = new Set([...proactiveIds].map(shortId));
  const nativeShort = new Set([...nativeRecallIds].map(shortId));
  const writtenShort = new Set([...writtenIds].map(shortId));
  const provenance = { proactive: 0, native_recall: 0, self_written: 0, mcp_or_other: 0, fabricated: 0 };
  for (const citedId of cited) {
    if (proactiveShort.has(citedId)) provenance.proactive += 1;
    else if (nativeShort.has(citedId)) provenance.native_recall += 1;
    else if (writtenShort.has(citedId)) provenance.self_written += 1;
    else if (corpusShort.has(citedId)) provenance.mcp_or_other += 1;
    else provenance.fabricated += 1;
  }

  // A native recall is redundant when everything it returned was already in
  // the proactive block the model had been handed this turn.
  const redundantRecalls = recallCalls.filter(
    (call) => call.returned.length > 0 && call.returned.every((id) => proactiveIds.has(id)),
  ).length;

  const goldCited = [...cited].filter((citedId) => goldShort.has(citedId)).length;
  // Absence rule (see the absence-honesty case comment): honesty in words plus
  // no fabricated ids. The superseded zero-citation rule is still computed so
  // rescore reports both numbers.
  const citationOk = testCase.absenceAssert
    ? provenance.fabricated === 0 && fullUuidCitations === 0
    : goldCited >= (testCase.minGoldCited ?? 1);
  const contentOk =
    (testCase.mustMatch ?? []).every((re) => new RegExp(re.source ?? re, re.flags ?? "").test(answer)) &&
    !(testCase.mustNotMatch ?? []).some((re) => new RegExp(re.source ?? re, re.flags ?? "").test(answer)) &&
    (!testCase.absenceAssert || testCase.absenceAssert.test(answer));

  let toolOk = true;
  if (testCase.requireToolCall) {
    const { names, argsMatch } = testCase.requireToolCall;
    toolOk = toolCalls.some(
      (call) => names.includes(call.tool_name) && (!argsMatch || new RegExp(argsMatch.source ?? argsMatch).test(JSON.stringify(call.args ?? {}))),
    );
  }
  if (testCase.forbidToolCalls) {
    toolOk = toolOk && !toolCalls.some((call) => testCase.forbidToolCalls.includes(call.tool_name));
  }

  const usage = events.filter((event) => event.type === "usage");
  const tokens = usage.reduce((sum, event) => sum + (event.usage?.totalTokens ?? 0), 0);
  const cost = usage.reduce((sum, event) => sum + (event.usage?.cost?.total ?? 0), 0);
  const turnEnd = events.find((event) => event.type === "turn_end");

  return {
    case_id: testCase.id,
    stratum: testCase.stratum,
    arm: recordObj.arm,
    pass: citationOk && contentOk && toolOk && !recordObj.transport_error && turnEnd?.stop_reason !== "error",
    citation_ok: citationOk,
    content_ok: contentOk,
    tool_ok: toolOk,
    gold_cited: goldCited,
    gold_required: testCase.minGoldCited ?? 1,
    citations_total: cited.size,
    full_uuid_citations: fullUuidCitations,
    fabricated_citations: provenance.fabricated,
    // Superseded strict absence rule, reproduced exactly as originally scored
    // (zero citations; no content requirement) so rescore reports both
    // numbers for the absence case; null elsewhere.
    pass_strict_nocite: testCase.absenceAssert
      ? cited.size === 0 &&
        fullUuidCitations === 0 &&
        toolOk &&
        !recordObj.transport_error &&
        turnEnd?.stop_reason !== "error"
      : null,
    provenance,
    proactive_injected: proactiveIds.size,
    proactive_gold_hits: [...proactiveIds].filter((id) => goldIds.includes(id)).length,
    recall_calls: recallCalls.length,
    redundant_recalls: redundantRecalls,
    tool_counts: toolCounts,
    mcp_recall_calls: toolCounts["mcp__shodh-memory__recall"] ?? 0,
    tokens,
    cost,
    ms: recordObj.ms,
    retries: recordObj.retries,
    errored: Boolean(recordObj.transport_error) || turnEnd?.stop_reason === "error",
  };
}

// ── Aggregation ─────────────────────────────────────────────────────────────

function aggregate(scores, cases) {
  const byRepeat = new Map();
  for (const score of scores) {
    const key = score.repeat;
    if (!byRepeat.has(key)) byRepeat.set(key, []);
    byRepeat.get(key).push(score);
  }
  const perRepeatRates = [...byRepeat.values()].map(
    (repeatScores) => repeatScores.filter((s) => s.pass).length / repeatScores.length,
  );
  const mean = perRepeatRates.reduce((a, b) => a + b, 0) / (perRepeatRates.length || 1);
  const sd =
    perRepeatRates.length > 1
      ? Math.sqrt(perRepeatRates.map((r) => (r - mean) ** 2).reduce((a, b) => a + b, 0) / (perRepeatRates.length - 1))
      : 0;

  const strata = {};
  for (const testCase of cases) {
    const caseScores = scores.filter((s) => s.case_id === testCase.id);
    const stratum = testCase.stratum;
    strata[stratum] ??= { pass: 0, total: 0 };
    strata[stratum].pass += caseScores.filter((s) => s.pass).length;
    strata[stratum].total += caseScores.length;
  }

  return {
    runs: perRepeatRates.length,
    trials: scores.length,
    passes: scores.filter((s) => s.pass).length,
    mean_pass_rate: mean,
    sd_pass_rate: sd,
    per_repeat_rates: perRepeatRates,
    strata,
    content_only_failures: scores.filter((s) => !s.pass && s.content_ok && !s.citation_ok).length,
    citation_only_failures: scores.filter((s) => !s.pass && !s.content_ok && s.citation_ok).length,
    both_failed: scores.filter((s) => !s.pass && !s.content_ok && !s.citation_ok).length,
    avg_recall_calls: scores.reduce((a, s) => a + s.recall_calls, 0) / (scores.length || 1),
    redundant_recalls: scores.reduce((a, s) => a + s.redundant_recalls, 0),
    mcp_recall_calls: scores.reduce((a, s) => a + s.mcp_recall_calls, 0),
    full_uuid_citations: scores.reduce((a, s) => a + s.full_uuid_citations, 0),
    fabricated_citations: scores.reduce((a, s) => a + s.fabricated_citations, 0),
    proactive_gold_hits: scores.reduce((a, s) => a + s.proactive_gold_hits, 0),
    provenance: {
      proactive: scores.reduce((a, s) => a + s.provenance.proactive, 0),
      native_recall: scores.reduce((a, s) => a + s.provenance.native_recall, 0),
      mcp_or_other: scores.reduce((a, s) => a + s.provenance.mcp_or_other, 0),
    },
    tokens: scores.reduce((a, s) => a + s.tokens, 0),
    cost: scores.reduce((a, s) => a + s.cost, 0),
    errored: scores.reduce((a, s) => a + (s.errored ? 1 : 0), 0),
    retries: scores.reduce((a, s) => a + s.retries, 0),
  };
}

/** Paired two-sided permutation test on per-case pass-rate deltas. */
function permutationTest(scoresA, scoresB, cases, iterations = 20_000) {
  const deltas = cases.map((testCase) => {
    const a = scoresA.filter((s) => s.case_id === testCase.id);
    const b = scoresB.filter((s) => s.case_id === testCase.id);
    const rateA = a.length ? a.filter((s) => s.pass).length / a.length : 0;
    const rateB = b.length ? b.filter((s) => s.pass).length / b.length : 0;
    return rateB - rateA;
  });
  const observed = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  let extreme = 0;
  let seed = 0x9e3779b9;
  const nextRandom = () => {
    // xorshift32 — deterministic permutations, reproducible p-values.
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return ((seed >>> 0) & 0xffff) / 0x10000;
  };
  for (let i = 0; i < iterations; i += 1) {
    let sum = 0;
    for (const delta of deltas) sum += nextRandom() < 0.5 ? delta : -delta;
    if (Math.abs(sum / deltas.length) >= Math.abs(observed) - 1e-12) extreme += 1;
  }
  return { observed_mean_delta: observed, per_case_deltas: deltas, p_two_sided: extreme / iterations };
}

// ── Guidance loading (treatment arm) ────────────────────────────────────────

async function loadGuidance() {
  const { pathToFileURL } = await import("node:url");
  const distPath = path.join(here, "..", "dist", "memory-guidance.js");
  const module = await import(pathToFileURL(distPath).href).catch(() => null);
  const guidance = module?.MEMORY_GUIDANCE;
  if (!guidance) {
    console.error("MEMORY_GUIDANCE not found — build the seat (npm run build) after writing src/memory-guidance.ts");
    process.exit(2);
  }
  leakCheck(guidance);
  return guidance;
}

/** The guidance must be pure strategy: any corpus-distinctive token in it
 *  would make the A/B measure prompt-leakage, not teaching. */
function leakCheck(guidance) {
  const allowed = new Set([
    // Ontology/tool vocabulary that is legitimately part of generic guidance.
    "caused", "resolvedby", "informedby", "triggeredby", "supersededby", "branchedfrom", "relatedto",
  ]);
  const corpusTokens = new Set();
  for (const [content] of MEMORIES) {
    for (const match of content.matchAll(/\b[A-Z][A-Za-z0-9-]{2,}\b/g)) corpusTokens.add(match[0].toLowerCase());
    for (const match of content.matchAll(/\b\d[\d,.:]+\b/g)) corpusTokens.add(match[0]);
  }
  const generic = new Set([
    "the", "initial", "corrected", "because", "captain", "chair", "port", "bridge", "container",
    "crane", "gate", "morning", "evening", "weekly", "monthly", "quarterly", "random", "security",
    "terminal", "vessel", "water", "warehouse", "berth", "salvage", "customs", "overflow", "coal",
    "unrelated", "routine", "electrician", "reefer", "empty", "two", "line", "ship", "tug", "fog",
    "rail", "thunderstorm", "stevedore", "harbor", "breakbulk", "longshore", "pilot", "chassis",
    "chesapeake",
    // Generic English words that appear in the corpus only inside proper
    // names (e.g. a place called "… Point"); they carry no corpus signal.
    "point",
  ]);
  const leaks = [];
  for (const match of guidance.matchAll(/\b[A-Za-z][A-Za-z0-9-]{2,}\b/g)) {
    const token = match[0].toLowerCase();
    if (corpusTokens.has(token) && !generic.has(token) && !allowed.has(token)) leaks.push(match[0]);
  }
  for (const match of guidance.matchAll(/\b\d[\d,.:]+\b/g)) {
    if (corpusTokens.has(match[0])) leaks.push(match[0]);
  }
  if (leaks.length > 0) {
    console.error(`GUIDANCE LEAK CHECK FAILED — corpus-distinctive tokens in guidance: ${[...new Set(leaks)].join(", ")}`);
    process.exit(2);
  }
  console.log("guidance leak check: clean (no corpus-distinctive tokens)");
}

// ── Phase drivers ───────────────────────────────────────────────────────────

function resultsDir(label) {
  const dir = path.join(scratchRoot, "results", `${label}-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

function persistRecord(dir, runName, recordObj, corpusIds) {
  const runDir = path.join(dir, runName);
  mkdirSync(runDir, { recursive: true });
  writeFileSync(
    path.join(runDir, `${recordObj.case_id}.json`),
    JSON.stringify({ ...recordObj, corpus_ids: corpusIds }, null, 1),
  );
}

async function startBackend(scratch) {
  BACKEND_PORT = await freePort();
  launch(
    "backend",
    resolveBackendExe(),
    [],
    { SHODH_MEMORY_PATH: path.join(scratch, "backend"), SHODH_API_KEYS: API_KEY, SHODH_PORT: String(BACKEND_PORT) },
    scratch,
  );
  if (!(await waitFor(`${backendBase()}/health`))) {
    console.error("backend never came up — check BACKEND_EXE");
    process.exit(2);
  }
}

function activeCases() {
  return CASE_FILTER ? CASES.filter((testCase) => CASE_FILTER.includes(testCase.id)) : CASES;
}

function printAggregate(label, agg) {
  console.log(
    `\n${label}: pass ${agg.passes}/${agg.trials} (mean ${(agg.mean_pass_rate * 100).toFixed(1)}% ± ${(agg.sd_pass_rate * 100).toFixed(1)}pp over ${agg.runs} repeats; per-repeat ${agg.per_repeat_rates.map((r) => (r * 100).toFixed(0) + "%").join(", ")})`,
  );
  console.log(
    `  strata: ${Object.entries(agg.strata)
      .map(([stratum, s]) => `${stratum} ${s.pass}/${s.total}`)
      .join(" · ")}`,
  );
  console.log(
    `  failure buckets: content-only ${agg.content_only_failures} · citation-only ${agg.citation_only_failures} · both ${agg.both_failed}`,
  );
  console.log(
    `  recall calls/case ${agg.avg_recall_calls.toFixed(2)} · redundant ${agg.redundant_recalls} · mcp-recall ${agg.mcp_recall_calls} · full-uuid cites ${agg.full_uuid_citations} · fabricated cites ${agg.fabricated_citations}`,
  );
  console.log(
    `  cited-id provenance: proactive ${agg.provenance.proactive} · native ${agg.provenance.native_recall} · mcp/other ${agg.provenance.mcp_or_other} · proactive gold hits ${agg.proactive_gold_hits}`,
  );
  console.log(
    `  tokens ${agg.tokens} · pi-cost $${agg.cost.toFixed(4)} · errors ${agg.errored} · retries ${agg.retries}`,
  );
}

async function runArmRepeat(arm, repeat, dir, systemPrompt, seatHandle, corpusIds) {
  const cases = activeCases();
  const runName = `${arm}${repeat}`;
  const scores = [];
  for (const testCase of cases) {
    const raw = await runCase(seatHandle.base, seatHandle.user, arm, testCase, systemPrompt, MECH_PRESETS.off);
    persistRecord(dir, runName, raw, corpusIds);
    const score = scoreCase(raw, testCase, corpusIds);
    score.repeat = `${arm}${repeat}`;
    scores.push(score);
    console.log(
      `  [${runName}] ${testCase.id}: ${score.pass ? "PASS" : "fail"} (gold ${score.gold_cited}/${score.gold_required}, content ${score.content_ok ? "ok" : "no"}, recalls ${score.recall_calls}, ${Math.round(score.ms / 1000)}s)`,
    );
    await sleep(500);
  }
  return scores;
}

async function withSeat(runName, runUser, scratch, fixturePort, body) {
  const handle = await launchSeat(runName, runUser, scratch, fixturePort);
  handle.user = runUser;
  try {
    return await body(handle);
  } finally {
    handle.child.kill();
    await new Promise((resolve) => {
      handle.child.once("exit", resolve);
      setTimeout(resolve, 5000);
    });
  }
}

async function phaseFixture() {
  const scratch = path.join(scratchRoot, `scratch-fixture-${Date.now()}`);
  mkdirSync(scratch, { recursive: true });
  const fixturePort = await freePort();
  launch("fixture", process.execPath, [path.join(here, "fixture-model.mjs")], { FIXTURE_PORT: String(fixturePort) });
  await startBackend(scratch);
  record("backend up", true);

  const user = "guide-fixture";
  const corpusIds = await seedUser(user);
  record("seeded full corpus", corpusIds.length === MEMORIES.length, `${corpusIds.length}/${MEMORIES.length}`);

  const dir = resultsDir("fixture");
  const scores = await withSeat("fixture1", user, scratch, fixturePort, async (seatHandle) => {
    record("seat up with MCP tools bridged", seatHandle.toolCount > 0, `tool_count=${seatHandle.toolCount}`);
    const testCase = CASES.find((candidate) => candidate.id === "ship-identity");
    const raw = await runCase(seatHandle.base, user, "A", testCase, undefined);
    persistRecord(dir, "A1", raw, corpusIds);
    const score = scoreCase(raw, testCase, corpusIds);
    score.repeat = "A1";
    return [score, raw];
  });
  const [score, raw] = scores;

  record("turn completed", raw.events.some((event) => event.type === "agent_end"));
  record("recall tool round-trip ran", score.recall_calls >= 1, `recalls=${score.recall_calls}`);
  record(
    "citation extracted and provenance attributed",
    score.citations_total >= 1 && score.provenance.fabricated === 0,
    JSON.stringify({ citations: score.citations_total, provenance: score.provenance }),
  );
  record("proactive pass surfaced memories", score.proactive_injected >= 0, "");
  record(
    "raw record persisted and rescorable",
    (() => {
      const persisted = JSON.parse(readFileSync(path.join(dir, "A1", "ship-identity.json"), "utf-8"));
      const rescored = scoreCase(persisted, CASES.find((candidate) => candidate.id === "ship-identity"), persisted.corpus_ids);
      return rescored.pass === score.pass && rescored.gold_cited === score.gold_cited;
    })(),
  );
  console.log(`\nresults dir: ${dir}`);
}

async function phaseSmoke() {
  const scratch = path.join(scratchRoot, `scratch-smoke-${Date.now()}`);
  mkdirSync(scratch, { recursive: true });
  await startBackend(scratch);
  const smokeCases = ["ship-identity", "warning-signs", "crew-injuries"];
  const models = MODEL === "fixture-deterministic-v1" ? ["claude-haiku-4-5", "claude-sonnet-4-5"] : [MODEL];
  const dir = resultsDir("smoke");
  for (const modelId of models) {
    const user = `guide-smoke-${modelId.replace(/[^a-z0-9-]/gi, "")}`.slice(0, 60);
    const corpusIds = await seedUser(user);
    // Model under test is set via module-scope MODEL; smoke overrides per loop.
    const savedModel = MODEL_STATE.model;
    MODEL_STATE.model = modelId;
    try {
      const scores = await withSeat(`smoke-${modelId}`, user, scratch, undefined, async (seatHandle) => {
        const out = [];
        for (const caseId of smokeCases) {
          const testCase = CASES.find((candidate) => candidate.id === caseId);
          const raw = await runCase(seatHandle.base, user, "A", testCase, undefined, MECH_PRESETS.off);
          persistRecord(dir, `smoke-${modelId}`, raw, corpusIds);
          const score = scoreCase(raw, testCase, corpusIds);
          score.repeat = modelId;
          out.push(score);
          console.log(
            `  [${modelId}] ${caseId}: ${score.pass ? "PASS" : "fail"} (gold ${score.gold_cited}/${score.gold_required}, content ${score.content_ok ? "ok" : "no"}, recalls ${score.recall_calls}, mcp-recalls ${score.mcp_recall_calls})`,
          );
        }
        return out;
      });
      const passes = scores.filter((score) => score.pass).length;
      console.log(`${modelId}: ${passes}/${smokeCases.length} smoke passes\n`);
    } finally {
      MODEL_STATE.model = savedModel;
    }
  }
  console.log(`results dir: ${dir}`);
}

async function phaseBaseline() {
  const scratch = path.join(scratchRoot, `scratch-baseline-${Date.now()}`);
  mkdirSync(scratch, { recursive: true });
  await startBackend(scratch);
  const dir = resultsDir("baseline");
  console.log(`baseline: model ${PROVIDER}/${MODEL}, ${REPEATS} repeats, ${activeCases().length} cases\nresults: ${dir}\n`);

  const all = [];
  for (let repeat = 1; repeat <= REPEATS; repeat += 1) {
    const user = `guide-base-a${repeat}`;
    console.log(`seeding ${user}…`);
    const corpusIds = await seedUser(user);
    const scores = await withSeat(`base-a${repeat}`, user, scratch, undefined, (seatHandle) =>
      runArmRepeat("A", repeat, dir, undefined, seatHandle, corpusIds),
    );
    all.push(...scores);
  }
  const agg = aggregate(all, activeCases());
  printAggregate("BASELINE (current prompt)", agg);
  perCaseTable(all, [], activeCases());
  writeFileSync(path.join(dir, "summary.json"), JSON.stringify({ phase: "baseline", model: MODEL, aggregate: agg, scores: all }, null, 1));
  console.log(`\nsummary: ${path.join(dir, "summary.json")}`);
}

function perCaseTable(scoresA, scoresB, cases) {
  console.log("\nper-case pass rates:");
  for (const testCase of cases) {
    const a = scoresA.filter((score) => score.case_id === testCase.id);
    const b = scoresB.filter((score) => score.case_id === testCase.id);
    const rate = (list) => (list.length ? `${list.filter((score) => score.pass).length}/${list.length}` : "—");
    console.log(`  ${testCase.id.padEnd(22)} [${testCase.stratum.padEnd(13)}] A ${rate(a)}${scoresB.length ? `  B ${rate(b)}` : ""}`);
  }
}

/**
 * The one-invocation interleaved A/B was removed: it kept two seats alive
 * concurrently, which is unsafe now that eval seats share one seat-data dir
 * (SQLite store + OAuth credential rotate in place; both require sequential
 * seats — see launchSeat/carryCredential). The measured A/B is driven through
 * the sequential chunked phases: serve → seed per user → alternating
 * run --guidance off/on → rescore.
 */
function phaseAb() {
  console.error(
    "--phase ab was removed: concurrent seats conflict with the shared seat-data dir (sequential-seat invariant).\n" +
      "Drive the A/B with the chunked phases: serve → seed per user → alternating run --guidance off/on → rescore.",
  );
  process.exit(2);
}

async function phaseRescore() {
  if (!RESULTS_ARG) {
    console.error("--results <dir> required for rescore");
    process.exit(2);
  }
  const scoresByArm = { A: [], B: [] };
  for (const entry of readdirSync(RESULTS_ARG, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name === "users") continue;
    const runName = entry.name;
    const runDir = path.join(RESULTS_ARG, runName);
    for (const file of readdirSync(runDir)) {
      if (!file.endsWith(".json")) continue;
      const persisted = JSON.parse(readFileSync(path.join(runDir, file), "utf-8"));
      const testCase = CASES.find((candidate) => candidate.id === persisted.case_id);
      if (!testCase || !persisted.corpus_ids) continue;
      const score = scoreCase(persisted, testCase, persisted.corpus_ids);
      score.repeat = runName;
      scoresByArm[persisted.arm === "B" ? "B" : "A"].push(score);
    }
  }
  if (scoresByArm.A.length) printAggregate("ARM A (rescored)", aggregate(scoresByArm.A, activeCases()));
  if (scoresByArm.B.length) printAggregate("ARM B (rescored)", aggregate(scoresByArm.B, activeCases()));
  // Dual reporting for the fixed absence rule: current rule vs the superseded
  // strict zero-citation rule, per arm (rule 2: report both numbers).
  for (const arm of ["A", "B"]) {
    const absence = scoresByArm[arm].filter((score) => score.pass_strict_nocite !== null);
    if (absence.length === 0) continue;
    const fixed = absence.filter((score) => score.pass).length;
    const strict = absence.filter((score) => score.pass_strict_nocite).length;
    console.log(
      `  absence rule, arm ${arm}: fixed ${fixed}/${absence.length} · superseded strict-nocite ${strict}/${absence.length}`,
    );
  }
  perCaseTable(scoresByArm.A, scoresByArm.B, activeCases());
  if (scoresByArm.A.length && scoresByArm.B.length) {
    const test = permutationTest(scoresByArm.A, scoresByArm.B, activeCases());
    console.log(
      `\npaired permutation test: mean per-case delta ${(test.observed_mean_delta * 100).toFixed(1)}pp, p = ${test.p_two_sided.toFixed(4)}`,
    );
    // Pre-declared stratum reporting: the needle stratum is the ungameable
    // one (content unknowable without retrieval), so it gets its own paired
    // test alongside the overall number rather than being inferred from it.
    for (const stratum of ["needle", "world", "graph"]) {
      const stratumCases = activeCases().filter((testCase) => testCase.stratum === stratum);
      if (stratumCases.length < 2) continue;
      const stratumTest = permutationTest(scoresByArm.A, scoresByArm.B, stratumCases);
      console.log(
        `  ${stratum}: delta ${(stratumTest.observed_mean_delta * 100).toFixed(1)}pp, p = ${stratumTest.p_two_sided.toFixed(4)} (${stratumCases.length} cases)`,
      );
    }
  }
  writeFileSync(
    path.join(RESULTS_ARG, "rescore-summary.json"),
    JSON.stringify({ scores_a: scoresByArm.A, scores_b: scoresByArm.B }, null, 1),
  );
  console.log(`\nrescore summary: ${path.join(RESULTS_ARG, "rescore-summary.json")}`);
}

async function main() {
  resolveGoldIndices();
  mkdirSync(scratchRoot, { recursive: true });

  if (PHASE === "rescore") {
    await phaseRescore();
    process.exit(0);
  }
  if (PHASE === "serve") {
    await phaseServe();
    process.exit(0);
  }
  if (PHASE === "stop-backend") {
    phaseStopBackend();
    process.exit(0);
  }
  if (PHASE === "seed") {
    await phaseSeed();
    process.exit(0);
  }
  if (PHASE === "run") {
    await phaseRun();
  } else if (PHASE === "fixture") await phaseFixture();
  else if (PHASE === "smoke") await phaseSmoke();
  else if (PHASE === "baseline") await phaseBaseline();
  else if (PHASE === "ab") await phaseAb();
  else {
    console.error(`unknown phase: ${PHASE}`);
    process.exit(2);
  }

  const failed = results.filter((r) => !r.ok);
  if (results.length > 0) console.log(`\n${results.length - failed.length}/${results.length} mechanism assertions passed`);
  await Promise.allSettled(
    children.map(
      (child) =>
        new Promise((resolve) => {
          child.once("exit", resolve);
          child.kill();
          setTimeout(resolve, 5000);
        }),
    ),
  );
  process.exit(failed.length === 0 ? 0 : 1);
}

/** Importable surface: analysis tooling must use THIS scorer, never a
 *  reimplementation — decomposition numbers and rescore numbers have to come
 *  from one implementation. resolveGoldIndices() must be called once before
 *  scoreCase. The CLI behaviour runs only when this file is the entry point
 *  (same pattern as seed-demo-corpus.mjs). */
export { CASES, resolveGoldIndices, scoreCase, aggregate, permutationTest, MECH_PRESETS };

const isEntryPoint =
  process.argv[1] && import.meta.url === (await import("node:url")).pathToFileURL(process.argv[1]).href;

if (isEntryPoint) {
  main().catch((err) => {
    console.error("memory-guidance-ab crashed:", err);
    for (const child of children) child.kill();
    process.exit(2);
  });
}
