# Warehouse Robot Memory Demo

Interactive demo: two robots inspect a warehouse. Robot Alpha finds hazards. Robot Beta enters 14 days later and avoids them using cross-mission memory recall.

## Quick Start

```bash
# 1. Start shodh-memory server
shodh-memory-server serve

# 2. Seed missions (16 waypoints + 12 lineage edges + Hebbian reinforcement)
node seed.js --api-key <your-key>

# 3. Serve the demo
python3 -m http.server 8080

# 4. Open http://localhost:8080 and enter your API key
```

## What It Shows

### Cross-Mission Memory
Alpha finds a cracked floor (Aisle 3) and damaged racking (Aisle 5). Beta enters the same warehouse 14 days later. As Beta approaches each hazard zone, the recall API fires against Alpha's stored memories — real match scores, real content.

### Causal Lineage
Click any waypoint to see lineage edges: InformedBy (cyan), TriggeredBy (amber), Caused (orange). Cross-mission edges have arrows. Edge count shown in legend.

### Live Recall
Every waypoint triggers `/api/recall` against all robots. The recall panel shows both cross-mission matches (cyan) and same-mission matches (robot color) with score bars, tier, and importance.

### Corridor-Following Paths
Robots follow the main corridor (y=250) and turn into aisles. No diagonal teleportation through shelving.

## Keyboard Shortcuts

- **Space** — Play / Stop
- **Right Arrow** — Next step
- **R** — Reset

## Seed Data

| Mission | Robot | Age | Waypoints | Hazards |
|---------|-------|-----|-----------|---------|
| Alpha Inspection | spot-alpha | 14 days | 8 | Cracked floor (A3), damaged racking (A5) |
| Beta Check | spot-beta | 30 min | 8 | Recalled both hazards, zero incidents |

4 cross-mission + 8 intra-mission lineage edges. Alpha's floor crack reinforced 2x (Hebbian).

## API Endpoints Used

`/api/remember`, `/api/recall`, `/api/search/robotics`, `/api/lineage/edges`, `/api/lineage/link`, `/api/reinforce`, `/api/consolidate`, `/api/forget/age`
