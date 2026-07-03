# Direct server mode with systemd

This guide is for Linux users who want the Rust HTTP server to run as a
long-lived local service and have MCP or REST clients connect to it.

This is optional. The default `npx -y @shodh/memory-mcp` setup is still the
quickest path for Claude Code, Cursor, Claude Desktop, and other MCP clients.

## When to use this

Use direct server mode when you want:

- A single backend process supervised by systemd.
- Stable health checks and logs outside of any one MCP client.
- REST clients, shell helpers, or multiple local tools to share the same server.
- The backend lifecycle to be independent from the MCP bridge lifecycle.

## Command roles

- `shodh server` starts the HTTP API server, usually on `127.0.0.1:3030`.
- `shodh serve` starts the MCP stdio server.
- `@shodh/memory-mcp` is the npm MCP wrapper used by MCP clients.

Direct server mode runs `shodh server` separately. MCP and REST clients can then
connect to that server instead of owning the backend process themselves.

## Create a user systemd service

First find the installed binary path:

```bash
command -v shodh
```

Create `~/.config/systemd/user/shodh-memory.service`:

```ini
[Unit]
Description=Shodh Memory HTTP server
After=network.target

[Service]
Type=simple
Environment=SHODH_HOST=127.0.0.1
Environment=SHODH_PORT=3030
Environment=SHODH_MEMORY_PATH=%h/.local/share/shodh-memory
Environment=SHODH_DEV_API_KEY=local-dev-key
ExecStart=%h/.cargo/bin/shodh server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Replace `ExecStart=%h/.cargo/bin/shodh server` with the path returned by
`command -v shodh` if your binary is installed somewhere else.

Start the service:

```bash
systemctl --user daemon-reload
systemctl --user enable --now shodh-memory.service
systemctl --user status shodh-memory.service
```

Check the server directly:

```bash
curl -sS http://127.0.0.1:3030/health
```

View logs:

```bash
journalctl --user -u shodh-memory.service -f
```

Restart after configuration changes:

```bash
systemctl --user restart shodh-memory.service
```

## MCP client example

If you want an MCP client to use this separately supervised server, point the
native MCP bridge at the HTTP API:

```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "shodh",
      "args": ["serve"],
      "env": {
        "SHODH_API_URL": "http://127.0.0.1:3030",
        "SHODH_API_KEY": "local-dev-key",
        "SHODH_USER_ID": "local-agent"
      }
    }
  }
}
```

Use the full path to `shodh` if your MCP client does not inherit your shell
`PATH`.

## User IDs

The `user_id` in API request bodies, and the `SHODH_USER_ID` environment
variable used by MCP and hook commands, identify the memory namespace for the
calling user or agent. This value is not an API key, password, or authentication
token.

Choose a stable `user_id` such as `local-agent`, `alice`, or `robot-1`, and keep
using the same value for that client if you want it to see the same memories.

## REST client example

All `/api/*` endpoints require the `X-API-Key` header.

```bash
export SHODH_API_URL=http://127.0.0.1:3030
export SHODH_API_KEY=local-dev-key
export SHODH_USER_ID=local-agent

curl -sS -X POST "$SHODH_API_URL/api/remember" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d "{\"user_id\":\"$SHODH_USER_ID\",\"content\":\"Prefer direct server mode\",\"memory_type\":\"Decision\"}"

curl -sS -X POST "$SHODH_API_URL/api/recall" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d "{\"user_id\":\"$SHODH_USER_ID\",\"query\":\"direct server mode\"}"
```

## Production notes

For local-only use, keep `SHODH_HOST=127.0.0.1`.

Before binding to `0.0.0.0` or exposing the service to another machine:

- Use production mode with `SHODH_ENV=production`.
- Set strong values in `SHODH_API_KEYS`.
- Put the service behind TLS and normal network access controls.
- Avoid using development keys such as `local-dev-key`.
