#!/usr/bin/env python3
"""Detectors for the defect classes this codebase has actually shipped.

Every check here exists because the class was found in the wild, not because it
is a general code smell:

  flags        172 env flags, 85 of which nothing outside src/ ever sets.

               CAREFUL WITH THIS ONE. "Never set by anything checked in" is NOT
               "never measured" -- most of these carry a documented A/B in the
               source comment beside them, with a CI run id. The finding is
               narrower and different: those measurements are NOT REPRODUCIBLE.
               The evidence lives in commit prose and run ids that expire, at
               baselines from another era (recall@10 0.6976 and 0.7124 against
               today's 0.5268), with no committed arm to re-run and no
               regression guard. A first pass of this audit reported them as
               unmeasured, which was wrong.

               The sharp case is SHODH_PPR: its default-ON justification cites
               "funnel graph present% 51->76", and the funnel was independently
               established to have run without the neural NER.

  dormant      Two whole modules (context.rs, injection.rs) were found alive
               only via their own tests. Code with no production caller is not
               dead weight, it is a claim of capability the product does not
               make.

  tests        A test that cannot fail is invisible to a failing-test sweep.
               Tautological asserts hid an inert Hebbian layer for months.

  ordering     A float sort with no tie-break leaves equal elements in input
               order; when the input comes from a map and a truncation follows,
               which rows survive is decided by iteration order. That is how
               the hub degree cap was permanently deleting different edges on
               every ingest.

  escapes      Env vars that switch off a safety check. Each is a way to
               produce numbers that look normal and are not.

Run:  python scripts/audit.py [--class flags|dormant|tests|ordering|escapes]

Exit code is always 0. This reports; it does not gate. Turning any of these
into a hard gate is a separate decision that needs a clean baseline first --
`dormant` in particular has known false positives (see its note).
"""

import argparse
import io
import os
import re
import sys
from glob import glob

SEP = chr(92)


def load_rust(root="src"):
    out = {}
    for base, _, files in os.walk(root):
        for f in files:
            if f.endswith(".rs"):
                p = os.path.join(base, f).replace(SEP, "/")
                out[p] = io.open(p, encoding="utf-8", errors="replace").read()
    return out


def load_external():
    """Everything that could exercise a flag: workflows, scripts, docs, tests."""
    texts = []
    for pat in (".github/workflows/*.yml", "scripts/*", "tests/**/*", "*.md", "docs/**/*"):
        for p in glob(pat, recursive=True):
            if os.path.isfile(p):
                try:
                    texts.append(io.open(p, encoding="utf-8", errors="replace").read())
                except OSError:
                    pass
    return "\n".join(texts)


def fn_bodies(src):
    """(attrs, name, body) for every fn. Brace-matched, so nested blocks are kept."""
    for m in re.finditer(r"((?:#\[[^\]]+\]\s*)*)fn\s+(\w+)\s*\([^)]*\)[^{]*\{", src):
        i, depth = m.end(), 1
        while i < len(src) and depth:
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        yield m.group(1), m.group(2), src[m.end():i - 1]


def strip_test_mods(src):
    """Drop #[cfg(test)] modules so 'used only by its own tests' is detectable."""
    out, i = [], 0
    while True:
        m = re.search(r"#\[cfg\(test\)\]\s*mod\s+\w+\s*\{", src[i:])
        if not m:
            out.append(src[i:])
            return "".join(out)
        start = i + m.start()
        out.append(src[i:start])
        j, depth = i + m.end(), 1
        while j < len(src) and depth:
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
            j += 1
        i = j


# ---------------------------------------------------------------- flags

ENGINE = (
    "src/memory/", "src/graph_memory.rs", "src/handlers/recall.rs", "src/relevance.rs",
    "src/vector_db/", "src/embeddings/", "src/handlers/state.rs", "src/relation_typer.rs",
)


def audit_flags(rust, ext):
    joined = "\n".join(rust.values())
    flags = sorted(set(re.findall(r"SHODH_[A-Z0-9_]+", joined)))
    rows = []
    for flag in flags:
        default, where = "?", []
        for path, src in rust.items():
            if flag not in src:
                continue
            where.append(path)
            i = src.find(flag)
            while i != -1:
                window = src[i:i + 700]
                if "unwrap_or(true)" in window:
                    default = "ON"
                elif "unwrap_or(false)" in window and default == "?":
                    default = "OFF"
                i = src.find(flag, i + 1)
        engine = any(w.startswith(e) or w == e for w in where for e in ENGINE)
        # ASSIGNED, not merely mentioned. Counting mentions lets documentation
        # silence the detector: adding this flag's name to a workflow input
        # description made it drop off its own list, which is a false negative
        # introduced by writing about it.
        assigned = len(re.findall(r"%s\s*[:=]" % re.escape(flag), ext))
        rows.append((flag, default, assigned, engine, where))

    never = [r for r in rows if r[2] == 0]
    on_never = [r for r in never if r[1] == "ON" and r[3]]

    print("## flags — %d total" % len(rows))
    print("   %d never set by any workflow, script or test." % len(never))
    print()
    print("   DEFAULT-ON BEHAVIOURAL FLAGS NOT EXERCISED BY ANYTHING CHECKED IN (%d):" % len(on_never))
    print("   Live in production. Most carry an A/B in the source comment beside")
    print("   them -- read it before assuming unmeasured. What is missing is a")
    print("   re-runnable arm, so nothing would catch a regression.")
    for flag, _, _, _, where in sorted(on_never):
        print("     %-36s %s" % (flag, where[0] if where else ""))
    return len(on_never)


# ---------------------------------------------------------------- dormant

SKIP_NAMES = {"new", "default", "fmt", "from", "clone", "drop", "next", "eq", "hash"}


def audit_dormant(rust):
    prod = {p: strip_test_mods(s) for p, s in rust.items()}
    prod_text = "\n".join(prod.values())
    all_text = "\n".join(rust.values())

    by_module = {}
    for path, src in prod.items():
        for m in re.finditer(r"\n\s*pub(?:\(crate\))?\s+(?:async\s+)?fn\s+(\w+)", src):
            name = m.group(1)
            if name.startswith("_") or name in SKIP_NAMES:
                continue
            if len(re.findall(r"\b%s\b" % re.escape(name), prod_text)) <= 1:
                only_tests = len(re.findall(r"\b%s\b" % re.escape(name), all_text)) - 1
                by_module.setdefault(path, []).append((name, only_tests))

    total = sum(len(v) for v in by_module.values())
    print("## dormant — %d public fns with no caller outside #[cfg(test)]" % total)
    print("   NOTE: this crate is published, so a `pub fn` may be external API rather")
    print("   than dead. Treat as a list to explain, not a list to delete.")
    for path in sorted(by_module, key=lambda k: -len(by_module[k]))[:12]:
        print("     %-44s %d" % (path, len(by_module[path])))
    return total


# ---------------------------------------------------------------- tests

ASSERT = re.compile(r"\bassert\w*!|\bpanic!|\bunreachable!|\.unwrap\(\)|\.expect\(")
TAUTOLOGIES = [
    re.compile(r"assert!\(\s*true\s*[,)]"),
    re.compile(r"assert_eq!\(\s*([A-Za-z_][\w.]*)\s*,\s*\1\s*[,)]"),
    re.compile(r"assert!\(\s*\w+\.len\(\)\s*>=\s*0\s*[,)]"),
]


def audit_tests(rust):
    silent, taut, total = [], [], 0
    for path, src in rust.items():
        # A local fn whose body asserts is an assertion helper; calling it counts.
        helpers = {n for a, n, b in fn_bodies(src) if ASSERT.search(b)}
        for attrs, name, body in fn_bodies(src):
            if "#[test]" not in attrs and "#[tokio::test]" not in attrs:
                continue
            total += 1
            for pat in TAUTOLOGIES:
                if pat.search(body):
                    taut.append((path, name))
                    break
            if ASSERT.search(body) or "?;" in body:
                continue
            if any(re.search(r"\b%s\s*\(" % re.escape(h), body) for h in helpers):
                continue
            silent.append((path, name))

    print("## tests — %d test fns" % total)
    print("   no assertion at all: %d" % len(silent))
    for path, name in silent:
        print("     %-44s %s" % (path, name))
    print("   tautological assertion: %d" % len(taut))
    for path, name in taut:
        print("     %-44s %s" % (path, name))
    return len(silent) + len(taut)


# ---------------------------------------------------------------- ordering

def audit_ordering(rust):
    hits = []
    for path, src in rust.items():
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if not re.search(r"\.(sort_by|sort_unstable_by|select_nth_unstable_by)\b", line):
                continue
            window = "\n".join(lines[i:i + 10])
            if not re.search(r"partial_cmp|total_cmp", window):
                continue
            if ".then" in window:
                continue
            truncates = bool(re.search(r"\.truncate\(|\.take\(", window))
            hits.append((path, i + 1, truncates))

    print("## ordering — %d float sorts with no tie-break" % len(hits))
    print("   A stable sort keeps equal elements in input order. When that input")
    print("   came from a map and a truncation follows, which rows survive is")
    print("   decided by iteration order. TRUNCATES marks that combination.")
    for path, line, truncates in sorted(hits, key=lambda h: (not h[2], h[0])):
        print("     %-9s %s:%d" % ("TRUNCATES" if truncates else "", path, line))
    return sum(1 for h in hits if h[2])


# ---------------------------------------------------------------- nested

FLAG_LET = re.compile(
    r'let\s+(\w+)\s*(?::[^=]+)?=\s*std::env::var\("(SHODH_[A-Z0-9_]+)"\)'
)
VAR_READ = re.compile(r'std::env::var\("(SHODH_[A-Z0-9_]+)"\)')


def audit_nested(rust):
    """Flags that are unreachable unless ANOTHER flag is already set.

    An arm that sets only the inner flag runs the default and reports a
    delta of zero, which is indistinguishable from the mechanism being
    inert -- so the flag looks measured and is not. Two have already
    shipped: SHODH_SPREAD_FIX, whose arms silently duplicated baseline, and
    SHODH_GRAPH_PATH_STATE, nested inside a SHODH_GRAPH_TRAVERSE that
    defaults OFF. The second cost a 1531-case run and would have been
    written down as the eighth inert graph lever.

    Nesting is not itself the defect -- an inner mechanism may only be
    meaningful inside the outer one, which is true of PATH_STATE. The
    defect is nesting it silently, because the prerequisite is invisible at
    the point where someone writes the arm, and because the CONTROL is then
    the outer flag alone rather than the default config. Comparing against
    the default charges the inner mechanism for the outer one's effect,
    which for TRAVERSE was -0.0320 recall.
    """
    hits = []
    for path, src in rust.items():
        idents = {m.group(1): m.group(2) for m in FLAG_LET.finditer(src)}
        for m in re.finditer(r"\bif\s+(\w+)\s*\{", src):
            outer = idents.get(m.group(1))
            if not outer:
                continue
            i, depth = m.end(), 1
            while i < len(src) and depth:
                if src[i] == "{":
                    depth += 1
                elif src[i] == "}":
                    depth -= 1
                i += 1
            body = src[m.end():i - 1]
            line = src[:m.start()].count("\n") + 1
            for inner in sorted(set(VAR_READ.findall(body))):
                if inner != outer:
                    hits.append((path, line, outer, inner))

    print("## nested -- %d flags reachable only when another flag is set" % len(hits))
    print("   An arm setting ONLY the inner flag measures baseline against")
    print("   baseline. Each needs its prerequisite documented at the read site,")
    print("   and its control is the OUTER flag alone -- never the default")
    print("   config, which confounds the two mechanisms.")
    for path, line, outer, inner in hits:
        print("     %-34s requires %-30s %s:%d" % (inner, outer, path, line))
    return len(hits)

# ---------------------------------------------------------------- escapes

def audit_escapes(rust, ext):
    joined = "\n".join(rust.values())
    names = sorted(set(re.findall(r"SHODH_(?:ALLOW|SKIP|DISABLE|FORCE)_[A-Z0-9_]+", joined)))
    print("## escapes — %d switches that disable a safety check" % len(names))
    print("   Each is a way to produce numbers that look normal and are not.")
    for n in names:
        where = [p for p, s in rust.items() if n in s]
        print("     %-40s used outside src/: %-3d  %s"
              % (n, ext.count(n), where[0] if where else ""))
    return len(names)


CHECKS = ("flags", "nested", "dormant", "tests", "ordering", "escapes")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="only", choices=CHECKS)
    args = ap.parse_args()

    rust = load_rust()
    ext = load_external()
    print("audited %d rust files, %d lines\n"
          % (len(rust), sum(s.count("\n") for s in rust.values())))

    run = [args.only] if args.only else CHECKS
    for name in run:
        {
            "flags": lambda: audit_flags(rust, ext),
            "nested": lambda: audit_nested(rust),
            "dormant": lambda: audit_dormant(rust),
            "tests": lambda: audit_tests(rust),
            "ordering": lambda: audit_ordering(rust),
            "escapes": lambda: audit_escapes(rust, ext),
        }[name]()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
