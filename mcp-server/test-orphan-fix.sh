#!/bin/bash
# Test script to verify MCP process orphan fix (#216)
# Run from mcp-server/ directory on macOS/Linux
#
# Tests:
# 1. Normal exit (SIGINT) — both processes should die
# 2. Stdin close — simulates MCP host session end
# 3. Multiple sequential sessions — no orphan accumulation
# 4. Reconnect timer doesn't prevent exit

set -e

PASS=0
FAIL=0
PORT=${SHODH_PORT:-3030}

log() { echo -e "\033[1;34m[TEST]\033[0m $1"; }
pass() { echo -e "\033[1;32m[PASS]\033[0m $1"; PASS=$((PASS + 1)); }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $1"; FAIL=$((FAIL + 1)); }

# Kill any existing shodh processes
pkill -f shodh-memory-server 2>/dev/null || true
pkill -f "node.*dist/index.js" 2>/dev/null || true
sleep 1

count_shodh_procs() {
  pgrep -f shodh-memory-server 2>/dev/null | wc -l | tr -d ' '
}

count_mcp_procs() {
  pgrep -f "node.*dist/index.js" 2>/dev/null | wc -l | tr -d ' '
}

# ─────────────────────────────────────────────
# Test 1: SIGINT cleanup
# ─────────────────────────────────────────────
log "Test 1: SIGINT cleanup"

# Start MCP server with stdin from /dev/null (will close immediately via stdin handler)
# But first test SIGINT path
echo '{"jsonrpc":"2.0","method":"initialize","params":{"capabilities":{}},"id":1}' | timeout 10 node dist/index.js 2>/dev/null &
MCP_PID=$!
sleep 3

# Check that server started
SERVER_COUNT=$(count_shodh_procs)
if [ "$SERVER_COUNT" -ge 1 ]; then
  log "  Server process running (count=$SERVER_COUNT)"
else
  fail "Test 1: Server process did not start"
fi

# Send SIGINT
kill -INT $MCP_PID 2>/dev/null || true
sleep 3

# Check both processes are gone
MCP_COUNT=$(count_mcp_procs)
SERVER_COUNT=$(count_shodh_procs)
if [ "$MCP_COUNT" -eq 0 ] && [ "$SERVER_COUNT" -eq 0 ]; then
  pass "Test 1: SIGINT killed both MCP and server processes"
else
  fail "Test 1: Orphan processes remain (MCP=$MCP_COUNT, Server=$SERVER_COUNT)"
  pkill -f shodh-memory-server 2>/dev/null || true
  pkill -f "node.*dist/index.js" 2>/dev/null || true
  sleep 1
fi

# ─────────────────────────────────────────────
# Test 2: Stdin close (simulates MCP host session end)
# ─────────────────────────────────────────────
log "Test 2: Stdin close (pipe EOF)"

# Use a subshell that closes stdin after a delay
(sleep 5; echo '{}') | timeout 15 node dist/index.js 2>/dev/null &
PIPE_PID=$!
sleep 8  # wait for stdin to close and process to react

MCP_COUNT=$(count_mcp_procs)
SERVER_COUNT=$(count_shodh_procs)
if [ "$MCP_COUNT" -eq 0 ]; then
  pass "Test 2: MCP process exited after stdin close"
else
  fail "Test 2: MCP process still alive after stdin close (count=$MCP_COUNT)"
  pkill -f "node.*dist/index.js" 2>/dev/null || true
fi

# Clean up server for next test
pkill -f shodh-memory-server 2>/dev/null || true
sleep 1

# ─────────────────────────────────────────────
# Test 3: No orphan accumulation over 3 sessions
# ─────────────────────────────────────────────
log "Test 3: Sequential sessions — no orphan accumulation"

for i in 1 2 3; do
  log "  Session $i starting..."
  (sleep 3; echo '{}') | timeout 10 node dist/index.js 2>/dev/null &
  sleep 6  # wait for stdin close + cleanup
done

sleep 2
MCP_COUNT=$(count_mcp_procs)
SERVER_COUNT=$(count_shodh_procs)

if [ "$MCP_COUNT" -eq 0 ]; then
  pass "Test 3: No orphaned MCP processes after 3 sessions (count=$MCP_COUNT)"
else
  fail "Test 3: $MCP_COUNT orphaned MCP processes after 3 sessions"
fi

# Server may still be running from last session (that's ok if only 1)
if [ "$SERVER_COUNT" -le 1 ]; then
  pass "Test 3: At most 1 server process (count=$SERVER_COUNT)"
else
  fail "Test 3: $SERVER_COUNT server processes (should be <= 1)"
fi

# Final cleanup
pkill -f shodh-memory-server 2>/dev/null || true
pkill -f "node.*dist/index.js" 2>/dev/null || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results: $PASS passed, $FAIL failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit $FAIL
