#!/usr/bin/env python3
"""Load Claude session data into shodh-memory.

Handles two formats:
  - session-*.json: Full session export with messages array
  - *.jsonl: Claude Code JSONL stream (line-delimited events)

Extracts assistant reasoning, thinking blocks, and text responses.
Loads as memories with original timestamps preserved.

Usage:
    python3 scripts/load_sessions.py <path> [<path>...] [options]
    python3 scripts/load_sessions.py /path/to/sessions/ --dry-run
    python3 scripts/load_sessions.py session.json *.jsonl --user-id autonomites-pipeline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def extract_from_session_json(path: str) -> list[dict]:
    """Extract memories from full session JSON (session-*.json format)."""
    with open(path) as f:
        data = json.load(f)

    memories = []
    session_id = data.get("session", {}).get("id", "unknown")[:8]

    for msg in data.get("messages", []):
        if msg.get("type") != "assistant":
            continue

        timestamp = msg.get("timestamp")
        content_blocks = msg.get("content", [])
        if isinstance(content_blocks, str):
            if len(content_blocks.strip()) > 50:
                memories.append({
                    "content": content_blocks.strip(),
                    "created_at": timestamp,
                    "tags": ["session", session_id],
                })
            continue

        _extract_content_blocks(content_blocks, timestamp, session_id, memories)

    return memories


def _is_autonomite_session(path: str) -> bool:
    """Check if a JSONL file is an autonomite session (not a regular Claude Code session).

    Autonomite sessions start with:
    - Pipeline run prompts: {"target_commit": ...}
    - Think phase: "You are in THINK MODE"
    - Subagent tasks: "<task-notification>"
    - Operator broadcasts: "## Operator Message"
    """
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get("type") == "queue-operation" and d.get("operation") == "enqueue":
                    content = d.get("content", "")[:300]
                    if not content:
                        return False
                    if content.strip().startswith("{") and "target_commit" in content:
                        return True
                    if "THINK MODE" in content:
                        return True
                    if "<task-notification>" in content:
                        return True
                    if "## Operator Message" in content:
                        return True
                    return False
            except json.JSONDecodeError:
                continue
    return False


def extract_from_jsonl(path: str) -> list[dict]:
    """Extract memories from JSONL stream (Claude Code session format)."""
    memories = []
    session_id = Path(path).stem[:8]

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Assistant messages contain the reasoning
            if event.get("type") == "assistant":
                msg = event.get("message", {})
                content_blocks = msg.get("content", [])
                timestamp = msg.get("timestamp") or event.get("timestamp")

                if isinstance(content_blocks, str):
                    if len(content_blocks.strip()) > 50:
                        memories.append({
                            "content": content_blocks.strip(),
                            "created_at": timestamp,
                            "tags": ["session", session_id],
                        })
                    continue

                _extract_content_blocks(content_blocks, timestamp, session_id, memories)

    return memories


def _is_process_noise(text: str) -> bool:
    """Detect agent process noise — tool mechanics, schema debugging, git ops.

    Returns True if the text is primarily about HOW the agent is working
    rather than WHAT it found. We want findings, not plumbing.
    """
    noise_signals = [
        # Tool/schema debugging
        "schema mismatch", "schema expects", "script processed 0",
        "script reports 0", "rewriting scratch", "let me check the script",
        "let me read the", "let me fix", "let me update",
        # Git mechanics
        "git commit", "git add", "git diff", "let me commit",
        "commit just my fixes", "commit the",
        # Playbook/meta process
        "updating the playbook", "update the playbook",
        "playbook updated", "mark these items as resolved",
        "now i need to", "now let me", "i'll start by reading",
        "let me analyze the", "let me look at the code",
        # Tool environment issues
        "sandbox blocked", "can't run selenium", "write tool seems restricted",
        "onnx runtime", "cargo check", "cargo test",
        # Autonomite design discussions (meta, not findings)
        "should graph-builder be", "approach a:", "approach b:",
        "i'd lean (a)", "premature abstraction",
    ]
    lower = text.lower()
    matches = sum(1 for s in noise_signals if s in lower)
    # If 2+ noise signals, it's process noise
    return matches >= 2


def _is_high_signal(text: str) -> bool:
    """Detect high-signal content — findings, observations, patterns.

    Returns True if the text contains domain-specific intelligence
    that would be useful for priming future autonomite runs.
    """
    signal_indicators = [
        # Domain findings
        "compromised", "injection", "redirect", "malicious",
        "gambling", "casino", "slot", "betting", "seo spam",
        # Infrastructure observations
        ".ac.th", ".ac.id", ".edu.", ".gov.",
        "wordpress", "liferay", "moodle", "ojs",
        "operator", "campaign", "cluster",
        # Analysis language
        "pattern", "finding", "discovered", "confirmed",
        "active", "cleaned", "dormant", "remediated",
        "critical", "high value", "productive",
        # Specific intelligence
        "whois", "registrar", "asn", "nameserver",
        "webshell", "cloaking", "waf", "cloudflare",
        "redirect chain", "hop", "final destination",
    ]
    lower = text.lower()
    matches = sum(1 for s in signal_indicators if s in lower)
    # 2+ signal indicators = likely high signal content
    return matches >= 2


def _extract_content_blocks(blocks: list, timestamp: str | None, session_id: str, memories: list):
    """Extract memories from content blocks (shared between formats).

    Filters for signal over noise: keeps findings, observations, and analysis.
    Skips tool debugging, schema fixes, git mechanics, and meta-process content.
    """
    text_parts = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "").strip()
            if text and not _is_process_noise(text):
                text_parts.append(text)

        elif block_type == "thinking":
            thinking = block.get("thinking", "").strip()
            if len(thinking) > 100 and not _is_process_noise(thinking):
                # Chunk long thinking, but only keep signal-rich chunks
                for i in range(0, len(thinking), 1500):
                    chunk = thinking[i:i + 1500].strip()
                    if len(chunk) > 100 and (_is_high_signal(chunk) or not _is_process_noise(chunk)):
                        memories.append({
                            "content": chunk,
                            "created_at": timestamp,
                            "memory_type": "observation",
                            "tags": ["thinking", session_id],
                        })

    # For text responses, keep summaries and findings, skip tool output narration
    combined = "\n".join(text_parts)
    if len(combined) > 50 and (_is_high_signal(combined) or len(combined) > 200):
        memories.append({
            "content": combined,
            "created_at": timestamp,
            "memory_type": "observation",
            "tags": ["response", session_id],
        })


def load_memories(memories: list[dict], api_url: str, api_key: str, user_id: str, dry_run: bool = False):
    """Load extracted memories into shodh-memory via API."""
    loaded = 0
    failed = 0

    for i, mem in enumerate(memories):
        payload = {
            "user_id": user_id,
            "content": mem["content"],
            "tags": mem.get("tags", []),
        }
        if mem.get("created_at"):
            payload["created_at"] = mem["created_at"]
        if mem.get("memory_type"):
            payload["memory_type"] = mem["memory_type"]

        if dry_run:
            preview = mem["content"][:80].replace("\n", " ")
            tags = ", ".join(mem.get("tags", []))
            print(f"  [{i+1}/{len(memories)}] [{tags}] {preview}...")
            loaded += 1
            continue

        try:
            req = Request(
                f"{api_url}/api/remember",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": api_key,
                },
                method="POST",
            )
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                loaded += 1
                if (i + 1) % 25 == 0:
                    print(f"  [{i+1}/{len(memories)}] loaded")
        except HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"  [{i+1}/{len(memories)}] FAILED {e.code}: {body[:100]}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"  [{i+1}/{len(memories)}] FAILED: {e}", file=sys.stderr)
            failed += 1

        # Throttle to avoid overwhelming the server on bulk load
        if not dry_run and (i + 1) % 10 == 0:
            time.sleep(0.05)

    return loaded, failed


def main():
    parser = argparse.ArgumentParser(description="Load Claude sessions into shodh-memory")
    parser.add_argument("sessions", nargs="+", help="Session JSON/JSONL files or directories")
    parser.add_argument("--api-url", default="http://localhost:3033")
    parser.add_argument("--api-key", default=None, help="API key (or set SHODH_API_KEY)")
    parser.add_argument("--user-id", default="autonomites-pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Preview without loading")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("SHODH_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: --api-key or SHODH_API_KEY required", file=sys.stderr)
        sys.exit(1)

    # Collect session files
    all_files = []
    for path in args.sessions:
        p = Path(path)
        if p.is_dir():
            all_files.extend(sorted(p.glob("session-*.json")))
            all_files.extend(sorted(p.glob("*.jsonl")))
        elif p.exists():
            all_files.append(p)
        else:
            print(f"Warning: {path} not found", file=sys.stderr)

    if not all_files:
        print("No session files found", file=sys.stderr)
        sys.exit(1)

    # Filter JSONL files to autonomite sessions only (skip regular Claude Code sessions)
    session_files = []
    skipped = 0
    for f in all_files:
        if f.name.endswith(".jsonl"):
            if _is_autonomite_session(str(f)):
                session_files.append(f)
            else:
                skipped += 1
        else:
            session_files.append(f)

    if args.limit > 0:
        session_files = session_files[:args.limit]

    print(f"Found {len(session_files)} autonomite sessions ({skipped} non-autonomite skipped)")

    total_loaded = 0
    total_failed = 0
    total_extracted = 0

    for session_file in session_files:
        name = session_file.name
        try:
            if name.endswith(".jsonl"):
                memories = extract_from_jsonl(str(session_file))
            elif name.endswith(".json"):
                memories = extract_from_session_json(str(session_file))
            else:
                continue

            total_extracted += len(memories)

            if memories:
                print(f"{name}: {len(memories)} memories")
                loaded, failed = load_memories(
                    memories, args.api_url, api_key or "", args.user_id, args.dry_run
                )
                total_loaded += loaded
                total_failed += failed
            else:
                print(f"{name}: 0 memories (skipped)")

        except json.JSONDecodeError as e:
            print(f"{name}: SKIP (invalid JSON: {e})", file=sys.stderr)
        except Exception as e:
            print(f"{name}: SKIP ({e})", file=sys.stderr)

    mode = "previewed" if args.dry_run else "loaded"
    print(f"\nDone: {total_extracted} extracted, {total_loaded} {mode}, {total_failed} failed "
          f"across {len(session_files)} files")


if __name__ == "__main__":
    main()
