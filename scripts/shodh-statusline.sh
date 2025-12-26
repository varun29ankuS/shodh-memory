#!/bin/bash
# Shodh Memory Status Line for Claude Code
# This script receives context info from Claude Code and:
# 1. POSTs to shodh-memory backend for TUI display
# 2. Outputs a formatted status line

# Configuration
SHODH_API="${SHODH_API_URL:-http://127.0.0.1:3030}"

# Read JSON from stdin
input=$(cat)

# Extract fields using jq
SESSION_ID=$(echo "$input" | jq -r '.session_id // "unknown"')
MODEL=$(echo "$input" | jq -r '.model.display_name // "Claude"')
CWD=$(echo "$input" | jq -r '.workspace.current_dir // .cwd // ""')
CONTEXT_SIZE=$(echo "$input" | jq -r '.context_window.context_window_size // 200000')
USAGE=$(echo "$input" | jq '.context_window.current_usage')

# Calculate context usage
if [ "$USAGE" != "null" ]; then
    INPUT_TOKENS=$(echo "$USAGE" | jq '.input_tokens // 0')
    CACHE_CREATE=$(echo "$USAGE" | jq '.cache_creation_input_tokens // 0')
    CACHE_READ=$(echo "$USAGE" | jq '.cache_read_input_tokens // 0')
    CURRENT_TOKENS=$((INPUT_TOKENS + CACHE_CREATE + CACHE_READ))
    PERCENT=$((CURRENT_TOKENS * 100 / CONTEXT_SIZE))
else
    CURRENT_TOKENS=0
    PERCENT=0
fi

# POST to shodh-memory backend (fire and forget, don't block status line)
curl -s -X POST "${SHODH_API}/api/context_status" \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"$SESSION_ID\", \"tokens_used\": $CURRENT_TOKENS, \"tokens_budget\": $CONTEXT_SIZE, \"current_dir\": \"$CWD\", \"model\": \"$MODEL\"}" \
    >/dev/null 2>&1 &

# Color codes based on usage
if [ "$PERCENT" -lt 50 ]; then
    COLOR="\033[32m"  # Green
elif [ "$PERCENT" -lt 80 ]; then
    COLOR="\033[33m"  # Yellow
else
    COLOR="\033[31m"  # Red
fi
RESET="\033[0m"

# Format token counts
if [ "$CURRENT_TOKENS" -gt 1000 ]; then
    TOKENS_FMT="$((CURRENT_TOKENS / 1000))k"
else
    TOKENS_FMT="$CURRENT_TOKENS"
fi

if [ "$CONTEXT_SIZE" -gt 1000 ]; then
    SIZE_FMT="$((CONTEXT_SIZE / 1000))k"
else
    SIZE_FMT="$CONTEXT_SIZE"
fi

# Extract just the last directory component for display
DIR_NAME=$(basename "$CWD" 2>/dev/null || echo "")

# Output status line (first line of stdout becomes the status)
echo -e "${COLOR}${PERCENT}%${RESET} ${TOKENS_FMT}/${SIZE_FMT} | ${MODEL} | ${DIR_NAME}"
