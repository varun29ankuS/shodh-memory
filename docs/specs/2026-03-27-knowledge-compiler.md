# Knowledge Compiler: Graph-Guided Runtime Knowledge Injection

## Problem

LLMs have two knowledge channels with a massive fidelity gap:

- **Training data**: Internalized, zero token cost, associative, always available. Feels like "knowing."
- **Prompt context**: Explicit text, burns tokens, linear, window-limited. Feels like "being told."

There's no runtime channel that delivers knowledge with training-data fidelity. Current workarounds (RAG, prompt stuffing, system prompt priming) all push text through the prompt-shaped hole. Even well-curated retrieval (compile_context with hybrid search, co-retrieval graphs, Hebbian decay) is still rendering knowledge as prose and hoping the model absorbs it.

## Core Insight

Separate **structure** from **content**.

The shodh memory graph has stable structural properties: Hebbian edge weighting, tiered decay (working -> session -> long-term), co-retrieval clustering, importance scoring. These define *how knowledge relates*. The actual memories define *what the knowledge is*.

The graph structure can guide a fast LoRA finetuning process: which memories to train on, how to batch them, how many epochs each gets, how to order the curriculum. The structure is the training plan. The content is the training data.

## Approaches (Tiered by Feasibility)

### Tier 1: Graph-Guided Fast Finetuning (Proven ML, Build Now)

Use the shodh graph topology to orchestrate standard LoRA finetuning on actual memory text. No hypernetwork, no novel architectures. Proven ML mechanics, applied intelligently.

```
BACKGROUND (minutes, on local GPU):
  shodh graph snapshot
    ├── memories: raw text + importance + tier + decay_state
    ├── edges: type + hebbian_weight + co_retrieval_count
    └── clusters: co-retrieval groups + aggregate importance
  → graph-guided curriculum builder
    ├── sampling probability from importance scores
    ├── training batches from co-retrieval clusters
    ├── epoch weighting from Hebbian edge strength
    └── cluster ordering from graph centrality
  → standard LoRA finetuning on memory text
  → adapter (~20-50MB)

AUTONOMITE SESSION:
  base model + LoRA adapter = model with internalized knowledge
```

**Why this works**: The graph doesn't generate weights — it generates a *training curriculum*. Which memories get more training steps (importance), which memories train together in the same batch (co-retrieval clusters), which associations the model needs to learn (high-Hebbian edges become synthetic QA pairs). Standard gradient descent does the rest.

**Graph-to-curriculum mapping:**

| Graph Feature | Curriculum Decision |
|---|---|
| Memory importance score | Sampling probability (high importance = more training steps) |
| Co-retrieval clusters | Batch composition (cluster members train together) |
| Hebbian edge weight | Synthetic association pairs ("X is related to Y because...") |
| Tier (working/session/longterm) | Epoch ordering (longterm first = stable foundation, working last = fresh context) |
| Decay state | Learning rate scaling (decayed memories get lower LR = softer encoding) |
| Graph centrality | Curriculum ordering (hub nodes train first, periphery inherits) |

**Synthetic training data from graph structure:**

The graph enables training data that raw text alone can't produce:

```
# From co-retrieval cluster:
Q: "What do you know about the UNS WordPress staff blog compromises?"
A: "Multiple staff blogs at *.staff.uns.ac.id are compromised with gambling
   SEO injection. Confirmed blogs: budi, dianrahmawati, daryono, harisudin.
   Cloaking evolved between runs — 100% googlebot_dual_trace by run 32,
   up from zero in run 31. The operator upgraded the redirect template
   mid-campaign."

# From Hebbian edge between two memories:
Q: "How does the UNS cluster relate to the broader Indonesian compromise pattern?"
A: "UNS is part of CLUSTER-ID-EDU-UNS-WP-SPAM, an Indonesian education sector
   campaign. It sits alongside go.id DNS subdomain takeovers (pn-jayapura,
   pa-mataram, bapenda.sumbawabaratkab) which represent a more severe
   compromise vector — DNS/hosting panel access vs. WordPress injection."
```

These synthetic QA pairs teach the model *associations* that exist in the graph but not in any single memory's text.

**Compute:**
- Local 7B model (Llama 3, Mistral): 10-30 minutes on consumer GPU (4090, M-series)
- Adapter size: 20-50MB
- Can run as background job while pipeline does other work
- Regenerate when graph changes significantly (staleness metric)

### Tier 2: Soft Prompt Compilation (Faster, Less Expressive)

Generate continuous prompt vectors (soft prompts) from graph features. Easier mapping problem than full LoRA generation — you're producing hundreds of virtual tokens, not millions of weight deltas.

```
EVERY RUN (seconds):
  graph snapshot → small encoder network → soft prompt vectors
  → prepend to model input (replaces text-based priming)
```

A soft prompt compiler is a much smaller model (~50-200M params) that learns to compress graph structure into a fixed-length sequence of continuous vectors. The target model treats these as "virtual tokens" that prime its attention state.

**Advantages over text priming:**
- Denser information per "token" (continuous vectors vs. discrete tokens)
- Compiler learns what the model responds to, not what reads well as prose
- Seconds to generate, not minutes to finetune

**Disadvantages:**
- Still consumes context positions (though fewer than text)
- Less precise than LoRA for exact factual recall
- Requires model-specific training (soft prompts aren't portable)

**Feasibility**: Moderate. Soft prompt tuning is well-established. The novel part is generating them from a graph structure via a learned compiler. Smaller research surface than Tier 3.

### Tier 3: Hypernetwork Zero-Shot Generation (Research Frontier)

A hypernetwork that takes a graph snapshot and outputs LoRA weights in a single forward pass. Maximum speed, maximum ambition, maximum research risk.

```
EVERY RUN (seconds):
  graph snapshot → hypernetwork → LoRA adapter
```

**Known challenges (from external review):**

1. **Lossy embedding problem**: Embeddings capture semantic neighborhoods, not exact strings. A 768d vector for "slot-toto.pn-jayapura.go.id" won't perfectly reconstruct that domain name. The hypernetwork needs a text pathway alongside the graph features — raw token sequences for memories that contain exact identifiers, embeddings for semantic content. Dual-pathway input.

2. **Output dimensionality**: Generating millions of LoRA parameters from a forward pass is an unstable optimization problem. Current hypernetwork successes (HyperNetworks for image generation, style transfer) operate on much smaller output spaces. Factual knowledge injection into LLM-scale adapters is unsolved.

3. **Bi-level optimization**: Training requires the target LLM in the loop — loss flows from LLM output, through the generated LoRA, back into the hypernetwork. Holding both in VRAM simultaneously and doing stable bi-level optimization at LLM scale is brutally expensive and fragile.

4. **Adapter capacity saturation**: A rank-16 LoRA has limited parameter budget. As memory count grows (1K -> 10K -> 100K), the adapter saturates. Need rank scaling, multi-adapter composition, or importance-based pruning of the graph before compilation.

**Feasibility**: Low near-term. This is publishable research, not engineering. Worth tracking as a moonshot but don't block anything on it.

## Input Representation (Shared Across Tiers)

All three tiers consume the same graph snapshot from shodh:

```
memory_node = {
  text: str,                     # raw content (for finetuning / text pathway)
  embedding: float[768],         # semantic vector (for soft prompts / hypernetwork)
  importance: float,             # current score after decay
  tier: enum(working|session|longterm),
  retrieval_count: int,          # times surfaced by compile_context
  age_hours: float,              # time since creation
  edge_count: int,               # degree in co-retrieval graph
  cluster_id: int,               # co-retrieval cluster membership
}

edge = {
  source: memory_id,
  target: memory_id,
  hebbian_weight: float,         # association strength
  co_retrieval_count: int,       # times retrieved together
  edge_type: enum(co_retrieval|tag_shared|temporal_proximity),
}

graph_snapshot = {
  memories: list[memory_node],
  edges: list[edge],
  clusters: list[{id, member_ids, aggregate_importance}],
  metadata: {total_memories, total_edges, graph_density, avg_importance},
}
```

The snapshot export is useful independently of the compiler — analysis, visualization, debugging the memory system.

## Integration with Shodh + Autonomites Pipeline

### Pipeline Flow (Future State)

```python
# In pipeline.py, before invoke_claude:

# 1. Check for pre-built adapter (background job keeps it fresh)
adapter_path = f"{workspace}/.adapters/{name}-latest.safetensors"

if os.path.exists(adapter_path) and adapter_is_fresh(adapter_path, shodh_mgr):
    # Tier 1: Use pre-built LoRA adapter with local model
    result = invoke_local_model(
        system_prompt,      # minimal — no memory section needed
        user_message,
        model="llama-3-70b",
        lora_adapter=adapter_path,
    )
elif shodh_mgr:
    # Fallback: prompt-based priming (current M0 approach)
    primed_context = shodh_mgr.compile_context(autonomite=name, ...)
    system_prompt = build_system_prompt(..., primed_context=primed_context)
    result = invoke_claude(system_prompt, user_message, ...)
```

### Background Adapter Builder

```python
# Runs as a background job, triggered by graph change events

def rebuild_adapter(shodh_mgr, autonomite, target_model, output_dir):
    snapshot = shodh_mgr.export_graph_snapshot(autonomite=autonomite)
    curriculum = build_curriculum(snapshot)  # graph -> training plan
    adapter = train_lora(
        base_model=target_model,
        training_data=curriculum.to_dataset(),
        rank=16,
        epochs=curriculum.recommended_epochs,
        lr_schedule=curriculum.lr_schedule,  # importance-weighted
    )
    adapter.save(f"{output_dir}/{autonomite}-latest.safetensors")
```

### Graceful Degradation

```
Has fresh adapter + local model?  → Tier 1: LoRA-equipped local inference
Has GPU + compiler trained?       → Tier 2: Soft prompt generation
Has shodh + API access?           → M0: Prompt-based priming (compile_context)
Nothing?                          → Raw playbook reading (original approach)
```

The retrieval and curation layer (shodh graph, compile_context, co-retrieval strengthening) is the same regardless of delivery format. Only the last mile changes.

### Cloud Model Path

Runtime adapter injection via cloud APIs faces a security barrier: providers are unlikely to execute arbitrary user-provided weights. More plausible models:

- **Provider-hosted finetuning**: Upload graph snapshot, provider runs the finetuning in their infrastructure, returns a model ID. Anthropic and OpenAI already offer finetuning APIs — the gap is making it fast enough for per-session use.
- **Cached adapter pools**: Pre-train adapters for common knowledge domains. Provider hosts them, user selects at inference time. Less flexible but avoids the arbitrary-weights problem.
- **Soft prompt injection**: If providers expose a continuous prompt prefix API (prepend these vectors to the KV cache), Tier 2 becomes viable for cloud models without the security concerns of weight injection.

### What Shodh Already Provides

| Shodh Feature | Compiler Input | Used By |
|---|---|---|
| Raw memory content | Training text for LoRA finetuning | Tier 1 |
| HNSW vector index | Pre-computed embeddings for soft prompts / hypernetwork | Tier 2, 3 |
| Hebbian edge weights | Synthetic association pairs, batch composition | Tier 1, 2, 3 |
| Co-retrieval graph | Cluster structure for grouped training / injection | All tiers |
| Tiered decay (exp -> power-law) | LR scaling, epoch ordering | Tier 1 |
| BM25 keyword index | Term-level boosting signals | Tier 1 |
| Importance scoring | Sampling probability, adapter capacity allocation | All tiers |

The graph *is* the curriculum. The compiler learns how to teach it.

## Compute Requirements

### Tier 1: Graph-Guided Finetuning
- **Per-build**: 10-30 minutes on consumer GPU (4090, M-series Apple Silicon)
- **Output**: LoRA adapter, 20-50MB for 7B, 50-200MB for 70B
- **Trigger**: On graph change (new runs, significant importance shifts)
- **Runs as**: Background job, doesn't block pipeline

### Tier 2: Soft Prompt Compiler
- **One-time training**: Days on 1-2 GPUs (much less than Tier 3)
- **Per-run generation**: Seconds (small encoder forward pass)
- **Output**: Soft prompt vectors, ~1-5MB
- **Runs as**: Inline at pipeline time (fast enough)

### Tier 3: Hypernetwork
- **One-time training**: Multi-day on 4+ A100s, bi-level optimization
- **Per-run generation**: Seconds (forward pass)
- **Output**: LoRA adapter, 20-50MB
- **Runs as**: Inline at pipeline time
- **Reality check**: Research project, not engineering project

## Milestones

### M0: Prompt-Based Priming (done)
compile_context renders memories as text in system prompt. Works today, burns tokens, good enough baseline.

### M1: Graph Snapshot Export
Add `shodh_mgr.export_graph_snapshot()` — serialize the memory graph into the shared input format defined above. Useful independently for analysis, visualization, and debugging.

### M2: Curriculum Builder
Implement the graph-to-curriculum mapping: importance-weighted sampling, co-retrieval batching, Hebbian synthetic QA generation. Output is a HuggingFace-compatible dataset. Validate that the curriculum "makes sense" before any finetuning.

### M3: Local Model PoC (Tier 1)
LoRA finetune a local 7B on the M2 curriculum. Evaluate:
- Does the model know facts from the graph without prompt priming?
- Does it know *associations* (co-retrieval clusters, Hebbian relationships)?
- Does importance weighting translate to recall priority?
- Does it still function as a general-purpose agent? (catastrophic forgetting check)
- A/B test: adapter-equipped local model vs. prompt-primed Claude on the same autonomite task.

### M4: Background Builder Integration
Wire the adapter builder into the pipeline as a background job. Auto-rebuild when graph staleness exceeds threshold. Local model runs get adapters automatically, cloud runs fall back to prompt priming.

### M5: Soft Prompt Exploration (Tier 2)
Train a small graph-to-soft-prompt encoder. Compare against Tier 1 on speed vs. fidelity tradeoff. If providers expose continuous prompt APIs, this becomes the cloud model path.

### M6: Hypernetwork Research (Tier 3)
Only if M3-M5 validate the core hypothesis that graph-guided knowledge injection outperforms prompt priming. Publish findings, contribute to the research direction.

## Open Questions

- **Adapter capacity**: How many memories can a rank-16 LoRA encode before saturation? The 1325-memory graph might fit easily; 100K might need rank scaling, multi-adapter composition, or importance-based graph pruning before compilation.
- **Catastrophic forgetting**: Does the adapter interfere with the base model's general capabilities? LoRA is generally safe here but needs validation on agent tasks, not just QA benchmarks.
- **Staleness threshold**: How much graph change justifies rebuilding the adapter? Need a delta metric — number of new memories, edge weight changes, importance shifts — to trigger background rebuilds efficiently.
- **Graph vs. flat**: Does the graph structure actually help compared to just finetuning on raw memory text in random order? The hypothesis is yes (associations transfer, curriculum ordering matters), but needs empirical validation at M3.
- **Evaluation methodology**: "The model knows this" vs. "the model can parrot this" — need tasks that require multi-hop reasoning across memories, not just single-fact recall. The co-retrieval clusters define natural multi-hop test cases.
- **Local model quality**: Can a 7B or 70B local model with a knowledge adapter match Claude's quality on domain-specific autonomite tasks? The adapter gives it knowledge, but the base model's reasoning capability still matters. May need to test across model sizes.
- **Provider API evolution**: Will cloud providers offer fast-turnaround finetuning, continuous prompt injection, or cached adapter pools? Track developments at Anthropic, OpenAI, Google. Any of these would unlock the cloud path.

## Prior Art and References

- **LoRA** (Hu et al., 2021): Low-rank adaptation. The adapter format for all tiers.
- **QLoRA** (Dettmers et al., 2023): Quantized LoRA. Enables finetuning large models on consumer GPUs.
- **Soft Prompt Tuning** (Lester et al., 2021): Learning continuous prompt vectors. Tier 2 foundation.
- **HyperNetworks** (Ha et al., 2016): Networks generating weights for other networks. Tier 3 foundation.
- **MAML** (Finn et al., 2017): Model-Agnostic Meta-Learning. The "learn to learn" framing.
- **Graph Neural Networks** (Kipf & Welling, 2016): GCN for encoding graph topology. Used in graph encoder across tiers.
- **Curriculum Learning** (Bengio et al., 2009): Training order matters. Tier 1's core principle.
