#!/usr/bin/env node
/**
 * corpus — a read-only document connector, over stdio.
 *
 * The first mile of the product, as a connector: point the seat at a folder of
 * documents and let the agent read them, so "ingest my corpus" is something an
 * analyst asks for in a sentence rather than a pipeline someone runs for them.
 *
 * Why this one, and why written here rather than borrowed:
 *
 *   It stays LOCAL. A stdio server is a process on this machine, so the egress
 *   badge keeps reading "Local" while the agent works. That matters more than
 *   it sounds: a badge that says "exits" for every connector teaches the
 *   operator to ignore it, and the claim is only worth making if it can also
 *   be seen holding.
 *
 *   It has no network dependency. Fetching someone's package at demo time is
 *   the wrong first move for a product sold on running with the cable out.
 *
 *   It is read-only BY CONSTRUCTION, not by policy. There is no write tool to
 *   withhold, so a misconfigured policy cannot make one appear. Policy is the
 *   second line of defence; not shipping the capability is the first.
 *
 * Every path is resolved and checked against the root before anything opens.
 * The check is on the RESOLVED path, because `../` in an argument is the
 * obvious attack and string-prefix checks on the unresolved path are the
 * obvious bug.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const ROOT = path.resolve(process.argv[2] ?? process.env.CORPUS_ROOT ?? ".");

/** Text this connector will open. Anything else is listed but not read: a
 *  binary read returns mojibake that looks like a corrupted document rather
 *  than like the wrong tool for the job. */
const READABLE = new Set([".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".log", ".xml", ".html"]);

/** Per-read cap. A model that asks for a 40MB log should get the head of it and
 *  a line saying so, not a context window full of one file. */
const MAX_BYTES = 200_000;

/** Guards against `../` and symlinks pointing out of the corpus. Resolves
 *  first, compares second — the other order is the bug. */
async function within(candidate) {
  const resolved = path.resolve(ROOT, candidate);
  const real = await fs.realpath(resolved).catch(() => resolved);
  const rootReal = await fs.realpath(ROOT).catch(() => ROOT);
  const rel = path.relative(rootReal, real);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error(`Path is outside the corpus root: ${candidate}`);
  }
  return real;
}

async function walk(dir, out = [], depth = 0) {
  if (depth > 6) return out;
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walk(full, out, depth + 1);
    else out.push(full);
  }
  return out;
}

const TOOLS = [
  {
    name: "list_documents",
    description:
      "List the documents in the corpus, with sizes. Start here: it tells you what is available " +
      "before you spend a read on the wrong file.",
    inputSchema: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Optional case-insensitive substring of the path." },
      },
    },
  },
  {
    name: "read_document",
    description:
      "Read one text document from the corpus. Returns the head of very large files with a note " +
      "saying so, rather than silently truncating.",
    inputSchema: {
      type: "object",
      properties: { path: { type: "string", description: "Path relative to the corpus root." } },
      required: ["path"],
    },
  },
  {
    name: "search_documents",
    description:
      "Find which documents contain a string, with the matching lines. Use before read_document " +
      "when you know what you are looking for but not where it is.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Case-insensitive substring." },
        max_hits: { type: "number", description: "Default 40." },
      },
      required: ["query"],
    },
  },
];

const server = new Server(
  { name: "corpus", version: "0.1.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;
  const text = (t) => ({ content: [{ type: "text", text: t }] });

  try {
    if (name === "list_documents") {
      const files = await walk(ROOT);
      const filtered = args.pattern
        ? files.filter((f) => f.toLowerCase().includes(String(args.pattern).toLowerCase()))
        : files;
      if (filtered.length === 0) return text("No documents match.");
      const rows = await Promise.all(
        filtered.sort().map(async (f) => {
          const { size } = await fs.stat(f);
          return `${path.relative(ROOT, f)}  ${size}B`;
        }),
      );
      return text(`${rows.length} document(s) under ${ROOT}:\n${rows.join("\n")}`);
    }

    if (name === "read_document") {
      const target = await within(String(args.path));
      if (!READABLE.has(path.extname(target).toLowerCase())) {
        return text(`Not a text document: ${args.path}. Readable extensions: ${[...READABLE].join(" ")}`);
      }
      const { size } = await fs.stat(target);
      const handle = await fs.open(target, "r");
      try {
        const buffer = Buffer.alloc(Math.min(size, MAX_BYTES));
        await handle.read(buffer, 0, buffer.length, 0);
        const body = buffer.toString("utf8");
        return text(
          size > MAX_BYTES
            ? `${body}\n\n[truncated: showing ${MAX_BYTES} of ${size} bytes]`
            : body,
        );
      } finally {
        await handle.close();
      }
    }

    if (name === "search_documents") {
      const needle = String(args.query).toLowerCase();
      const limit = Number(args.max_hits ?? 40);
      const files = (await walk(ROOT)).filter((f) => READABLE.has(path.extname(f).toLowerCase()));
      const hits = [];
      for (const file of files) {
        if (hits.length >= limit) break;
        const { size } = await fs.stat(file);
        if (size > MAX_BYTES * 5) continue;
        const content = await fs.readFile(file, "utf8").catch(() => "");
        content.split(/\r?\n/).forEach((line, i) => {
          if (hits.length < limit && line.toLowerCase().includes(needle)) {
            hits.push(`${path.relative(ROOT, file)}:${i + 1}: ${line.trim().slice(0, 200)}`);
          }
        });
      }
      return text(hits.length ? hits.join("\n") : `No document contains "${args.query}".`);
    }

    return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
  } catch (error) {
    // Returned as a tool error rather than thrown: a path outside the root is
    // something the model should see and correct, not a transport failure that
    // drops the connector for the rest of the session.
    return { content: [{ type: "text", text: String(error.message ?? error) }], isError: true };
  }
});

await server.connect(new StdioServerTransport());
console.error(`[corpus] read-only, rooted at ${ROOT}`);
