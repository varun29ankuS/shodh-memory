# Shodh Memory

## Cognitive Memory Infrastructure for Autonomous Systems

Varun Sharma | Founder
varun@shodh-memory.com | shodh-memory.com

---

# The Problem

## Autonomous systems have amnesia

Every robot, drone, and AI agent starts from zero. Every session.

- **Warehouse robot** repeats the same navigation mistakes daily
- **Agricultural drone** can't learn which fields need attention
- **Manufacturing arm** doesn't remember what worked yesterday

**The cost:** Slower operations, repeated errors, wasted compute

**Root cause:** Cloud memory adds latency and requires connectivity. Edge devices need memory that runs locally, learns continuously, and never forgets what matters.

---

# Why This Matters Now

## The edge AI inflection point

Three trends converging:

**1. Robots are shipping**
- 500K+ industrial robots deployed annually
- Warehouse AMRs growing 25% YoY
- Agricultural autonomy reaching commercial scale

**2. Cloud is the bottleneck**
- 100-500ms latency kills real-time control
- Connectivity isn't guaranteed in fields, factories, warehouses
- Data sovereignty concerns in defense, healthcare, manufacturing

**3. Memory is the missing piece**
- LLMs gave us reasoning
- Vector DBs gave us retrieval
- Nobody solved learning and retention at the edge

---

# The Solution

## Shodh Memory

A cognitive memory system that learns with use. Runs fully offline. Single ~30MB binary.

**Not a vector database.** A memory system modeled on how brains actually work.

- **3-tier architecture** (working → session → long-term) based on Cowan's cognitive model
- **Hebbian learning** — connections strengthen through use, fade through neglect
- **Knowledge graph** with spreading activation retrieval
- **Memory consolidation** with interference detection

**Result:** Systems that get better over time, without cloud dependency.

---

# How It Works

## Biologically-inspired memory architecture

```
Sensory Input → Working Memory → Session Memory → Long-Term Memory
                     ↓               ↓                ↓
                 (100 items)     (RocksDB)      (Consolidated)
                     ↓               ↓                ↓
              Hebbian Learning → Knowledge Graph → Semantic Facts
```

**Key mechanisms:**

| Mechanism | What it does |
|-----------|--------------|
| Hebbian plasticity | Frequently accessed memories strengthen permanently |
| Hybrid decay | Power-law forgetting (matches human memory research) |
| Consolidation | Old episodes compress to semantic facts |
| Spreading activation | Related memories surface automatically |

**Grounded in neuroscience:** Architecture informed by 18+ peer-reviewed cognitive science papers.

---

# Technical Specifications

## Built for edge constraints

| Specification | Value |
|---------------|-------|
| Binary size | ~30MB (single file, no dependencies) |
| Graph lookup | <1μs |
| Semantic search | 34-58ms |
| Memory footprint | ~50MB runtime |
| Storage | RocksDB (embedded, crash-safe) |
| Vector index | DiskANN/HNSW (billion-scale) |

**Platform support:**
- Linux x86_64, ARM64
- macOS (Intel, Apple Silicon)
- Windows x86_64
- Tested: Raspberry Pi 4/5, NVIDIA Jetson, Industrial PCs

**Stack:** Rust, ONNX Runtime, RocksDB, PyO3

---

# Why Edge & Robotics

## Cloud memory doesn't work for autonomous systems

| Requirement | Cloud Memory | Shodh Memory |
|-------------|--------------|--------------|
| Latency | 100-500ms | <1μs graph, 34-58ms semantic |
| Offline operation | No | Yes |
| Data sovereignty | Data leaves device | Stays on device |
| Deployment size | Requires infrastructure | ~30MB binary |
| Runs on | Servers | Raspberry Pi to Jetson |

**Target deployments:**
- Industrial robots and manufacturing cells
- Agricultural drones and autonomous tractors
- Warehouse automation (AMRs, pick-and-place)
- Defense systems (air-gapped, zero network)

---

# Current Traction

## AI agents as high-velocity testbed

Robotics deployments have longer sales cycles. AI coding agents provide fast iteration and real revenue today.

**Why AI agents first:**
- Developers push the system hard, daily
- Complex memory patterns surface quickly
- Revenue validates the core technology
- Same memory system deploys to robotics

**Metrics:**

| Metric | Value |
|--------|-------|
| GitHub | 36 stars, 7 forks |
| Tests | 688 passing |
| Releases | 8 versions shipped |
| Integrations | PyPI, npm (MCP), crates.io |
| Revenue | Early, from AI agent users |

---

# Market Opportunity

## Memory infrastructure for autonomous systems

**Primary market:** Edge AI and robotics

| Segment | TAM (2030) |
|---------|------------|
| Industrial robotics | $50B+ |
| Autonomous vehicles | $60B+ |
| Agricultural robotics | $20B+ |
| Warehouse automation | $30B+ |

**Go-to-market:**
1. **Now:** AI agent developers (fast feedback, revenue)
2. **Next:** Robotics pilots with design partners
3. **Scale:** Enterprise licensing for fleet deployments

**Business model:**
- Open source core (adoption, community)
- Enterprise licensing (robotics, manufacturing)
- Support contracts (mission-critical systems)

---

# Competitive Landscape

## Others store. Shodh learns.

| Capability | Shodh | Mem0 | Zep | Pinecone |
|------------|-------|------|-----|----------|
| Runs fully offline | Yes | No | No | No |
| Hebbian learning | Yes | No | No | No |
| Single binary (<50MB) | Yes | No | No | No |
| Runs on Raspberry Pi | Yes | No | No | No |
| Knowledge graph | Yes | No | Yes | No |
| Memory decay model | Yes | No | No | No |
| Sub-millisecond lookup | Yes | No | No | No |

**Moat:**
- Neuroscience-grounded architecture (not just vector search)
- Rust performance (memory-safe, sub-microsecond ops)
- Edge-first design (competitors are cloud-first, adapting down)

---

# Founder

## Varun Sharma

Mechanical engineer who builds physical systems. Now building memory for them.

| Experience | Relevance |
|------------|-----------|
| **MS Mechanical Engineering, Case Western Reserve** | Robotics, controls, systems thinking |
| **Sr. Project Scientist, IIT Delhi** | Flywheel energy storage, manufacturing R&D |
| **Faros Technologies** | 3DOF/6DOF motion platforms, led engineering team |
| **Reflow** | Manufacturing R&D, Six Sigma, circular economy |
| **Serial founder** | Karmath (SRM motors), Atulyam (supply chain), Mool |

**Why me:**
- Built motion platforms — I know what robots need
- Research background — can go deep on the science
- Hardware + software — concept to production, not just code

---

# The Ask

## Introductory conversation

Looking to discuss:

1. **Feedback** on robotics/edge positioning
2. **Introductions** to robotics companies in Stata portfolio
3. **Potential investment** as we scale edge deployments

**Use of funds:**
- Robotics pilot deployments with design partners
- Embedded optimization (ARM, RISC-V targets)
- First enterprise sales hire

---

# Thank You

## Let's talk

**Varun Sharma**
Founder, Shodh Memory

varun@shodh-memory.com
shodh-memory.com
github.com/varun29ankuS/shodh-memory

Case Western Reserve, MS Mechanical Engineering

---
