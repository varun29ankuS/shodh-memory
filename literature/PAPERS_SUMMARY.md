# Research Papers Summary

## AI Memory Systems

### 1. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
**ArXiv:** https://arxiv.org/abs/2504.19413

**Authors:** Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav

**Abstract Summary:**
Addresses the limitation of LLM fixed context windows for multi-session conversations. Mem0 dynamically extracts, consolidates, and retrieves salient information from ongoing conversations. Includes graph-based memory variant for complex relationships.

**Key Findings:**
- 26% relative improvements in LLM-as-a-Judge metric over OpenAI
- Graph-enhanced version adds ~2% additional improvement
- 91% lower p95 latency, >90% token cost savings vs full-context
- Outperformed 6 baseline categories on LOCOMO benchmark

**Relevance to Shodh:**
- Validates hybrid memory approach (vector + graph)
- Limitation: no linguistic/cognitive foundation

---

### 2. Zep: A Temporal Knowledge Graph Architecture for Agent Memory
**ArXiv:** https://arxiv.org/abs/2501.13956

**Authors:** Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef

**Abstract Summary:**
Memory service integrating dynamic knowledge from conversations and business data. Core component "Graphiti" is a temporally-aware knowledge graph engine.

**Key Findings:**
- DMR Benchmark: 94.8% accuracy (vs MemGPT's 93.4%)
- LongMemEval: up to 18.5% accuracy improvements on temporal reasoning
- 90% reduced response latency vs baseline

**Relevance to Shodh:**
- Shows temporal graphs capture context shifts well
- Limitation: entity-relationship based, not grammar-aware

---

### 3. MemGPT: Towards LLMs as Operating Systems
**ArXiv:** https://arxiv.org/abs/2310.08560

**Authors:** Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez

**Abstract Summary:**
Proposes "virtual context management" borrowing from OS memory hierarchies. Enables LLMs to work with extended context through intelligent memory tier management.

**Key Findings:**
- Effective for document analysis beyond context window
- Enables multi-session chat with memory persistence
- OS-inspired design for memory management

**Relevance to Shodh:**
- Tier-based memory concept aligns with our working/session/long-term model
- Limitation: no understanding of *what* to remember

---

## AI Memory Frameworks Survey

### Survey of AI Agent Memory Frameworks
**Source:** https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks

**Frameworks Compared:**

| Framework | Architecture |
|-----------|-------------|
| **Letta** | Layered memory: in-context, core blocks, archival with embedding retrieval |
| **Mem0.ai** | Conversation history + user preferences, multiple vector DB support |
| **Zep** | Session-based with knowledge graphs, automated summarization |
| **CrewAI** | Distributed across "crews", entity memory layers |
| **Memary** | Knowledge graph expansions, persistent memory, "rewind" capabilities |
| **Cognee** | Unified immediate context + external vector/graph databases |

**Common Gaps:**
- Automated compression
- Multi-agent orchestration
- Fully-featured local deployments

**Shodh Advantage:**
- 100% local/offline capability unique for robotics/edge AI

---

## Cognitive Psychology Research

### Working Memory and Cognitive Development
**Source:** [PMC4207727](https://pmc.ncbi.nlm.nih.gov/articles/PMC4207727/)

**Author:** Nelson Cowan (University of Missouri)

**Title:** Working Memory Underpins Cognitive Development, Learning, and Education

**Key Findings:**
- Working memory capacity limited to ~3-4 items at any time
- Distinction between activated long-term memory and focus of attention
- Working memory serves as infrastructure for binding concepts together
- Capacity increases through childhood via maturation, knowledge acquisition, and processing speed
- Strategic instruction adjustments support learning more than isolated WM training

**Implications for Shodh:**
- Validates our tiered memory model (working → session → long-term → archive)
- Supports limited "focus of attention" concept - not everything can be active
- Binding concepts together = graph relationships in Shodh
- Knowledge acquisition improves memory capacity = frequency-based salience
- Aligns with our approach: adjust what's in working memory based on task context

---

### Emotional Tagging Hypothesis
**Source:** PMC10410470

**Key Finding:**
"The activation of the amygdala in emotionally arousing events helps to mark experiences as necessary, thus enhancing synaptic plasticity and facilitating transformation from transient into more permanent forms for encoding long-term memories."

**Implications for Shodh:**
- High-arousal verbs (killed, discovered, failed) should boost memory importance
- Emotional significance determines what gets consolidated
- Provides cognitive basis for verb classification (Phase 3)

---

### Emotional Arousal and Memory Priority

**Key Concept:**
Arousal enhances memory for HIGH PRIORITY stimuli while IMPAIRING memory for LOW PRIORITY stimuli. Priority determined by:
- Bottom-up salience (inherent importance)
- Top-down goals (user-defined importance)

**Implications for Shodh:**
- Validates salience-based approach
- High-salience entities deserve more persistence
- Low-salience can be compressed/forgotten

---

### Central vs Peripheral Memory

**Key Finding:**
Violence/trauma improve memory for CENTRAL GIST while impairing memory for PERIPHERAL DETAILS.

**Implications for Shodh:**
- Store the gist (nouns/entities), not every detail
- Adjectives/adverbs are peripheral, can decay faster
- Supports our "compress low-salience" strategy

---

## Construction Grammar / Cognitive Linguistics

### Usage-based Grammar Induction
**Source:** MIT Computational Linguistics

**Key Findings:**
- Sequence memory + chunking are key cognitive mechanisms
- Memory stores "constructions" (chunks), not individual words
- Frequent patterns become deeply entrenched (high token frequency)
- Type frequency strengthens constructional schema
- High token frequency expressions resist change

**Implications for Shodh:**
- Frequent entity co-occurrences should strengthen graph edges
- Entity "constructions" (patterns) become more salient
- Validates frequency-based salience updates

---

## Key Takeaways for Gravitational Salience Memory

1. **Hybrid is best**: Vector + Graph outperforms single-store (Mem0)
2. **Temporal matters**: Time-aware graphs capture context (Zep)
3. **Tier management works**: OS-style memory layers effective (MemGPT)
4. **Local is unique**: No competitor offers 100% offline for edge AI
5. **Emotions tag importance**: Amygdala activation = salience marker
6. **Gist over details**: Central information preserved, peripheral decays
7. **Frequency entrenchment**: Repeated patterns become more important

## Shodh's Unique Position

First memory system grounded in cognitive linguistics:
- Nouns = gravitational wells (salience determines mass)
- Verbs = pathways (arousal determines importance boost)
- Adjectives = filters/modifiers
- Forgetting = salience-weighted, not arbitrary FIFO/LRU
