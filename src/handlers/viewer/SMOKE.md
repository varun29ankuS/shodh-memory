# Manual Smoke Test (run before migration flip)

1. Load a real shodh instance with >1k nodes at `/graph/view2?user_id=<you>`. Viewer opens, renders within 5s.
2. Trigger a burst of activity (50 recalls in a minute) via `shodh recall ...`. Pulses appear on active edges; no layout jitter.
3. Close and reopen mid-session. Reconnect works; state consistent.
4. Drag-drop a snapshot `.gexf` onto the canvas in drop-zone mode (`/graph/view2` with no `user_id`). Graph replaces contents.
