# GEXF Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a browser-based viewer for shodh's GEXF export that makes Hebbian memory dynamics (spreading activation, edge weights, LTP tier promotions) visually legible, served in-tree at `/graph/view2`.

**Architecture:** Fork `../viewer/` at commit `827c191`, strip HTTP-Graph/ISAP modules, vendor into `src/handlers/viewer/`. Load GEXF via existing export endpoint; live updates triggered by `/api/events/sse` (with new query-param auth fallback) + in-place graph-diff mutation preserving viewport/selection/FA2 state. Sigma.js + graphology + vanilla JS, no build step.

**Tech Stack:** Rust (axum, existing), sigma.js, graphology, graphology-gexf, graphology-layout-forceatlas2, Vitest + jsdom, Playwright.

**Spec:** `docs/specs/2026-04-16-gexf-viewer-design.md`

---

## Phase 1 — GEXF Export Additions (Backend Prerequisites)

Spec §"GEXF Export Additions". These expose attrs already on the internal model so the viewer's domain styling has data to key on.

### Task 1: Add new edge attrs to GEXF emission

Edges already emit `weight=` (strength), `ltp_status`, `tier`, `activation_count`. Add `last_activated`, `created_at`, `valid_at`, `entity_confidence`.

**Files:**
- Modify: `src/handlers/export.rs:299-304` (edge attribute declarations)
- Modify: `src/handlers/export.rs:383-401` (edge attvalue emission loop)
- Test: `src/handlers/export.rs` (inline `tests` module)

- [ ] **Step 1: Write the failing test**

Add to the existing `tests` module in `src/handlers/export.rs`:

```rust
#[test]
fn test_gexf_emits_new_edge_attributes() {
    let rel = make_relationship_edge();
    let response = GraphExportResponse {
        metadata: ExportMetadata {
            exported_at: Utc::now(),
            user_id: "u".into(),
            node_count: 0,
            edge_count: 1,
            node_counts_by_type: HashMap::new(),
            edge_counts_by_type: HashMap::new(),
        },
        nodes: vec![],
        edges: vec![relationship_to_edge(&rel)],
    };
    let gexf = to_gexf(&response);

    // Declarations
    assert!(gexf.contains(r#"title="last_activated""#));
    assert!(gexf.contains(r#"title="created_at""#));
    assert!(gexf.contains(r#"title="valid_at""#));
    assert!(gexf.contains(r#"title="entity_confidence""#));

    // Values (entity_confidence on make_relationship_edge() is 0.9)
    assert!(gexf.contains(r#"value="0.9""#), "entity_confidence value missing");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::export::tests::test_gexf_emits_new_edge_attributes -- --nocapture 2>&1 | tail -20`
Expected: FAIL — assertion on `title="last_activated"` fails.

- [ ] **Step 3: Add edge attribute declarations**

At `src/handlers/export.rs:299-304`, expand the `<attributes class="edge">` block:

```rust
writeln!(out, r#"    <attributes class="edge">"#).unwrap();
writeln!(out, r#"      <attribute id="0" title="type" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="1" title="ltp_status" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="2" title="tier" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="3" title="activation_count" type="integer"/>"#).unwrap();
writeln!(out, r#"      <attribute id="4" title="last_activated" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="5" title="created_at" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="6" title="valid_at" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="7" title="entity_confidence" type="float"/>"#).unwrap();
writeln!(out, r#"    </attributes>"#).unwrap();
```

GEXF `xsd:dateTime` has no standard GEXF type, so DateTimes are serialized as ISO-8601 strings.

- [ ] **Step 4: Emit the new attvalues**

After the existing `activation_count` block (around line 401 before `</attvalues>`), add:

```rust
// for="4" last_activated (ISO-8601 string)
if let Some(v) = edge.attributes.get("last_activated").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="4" value="{v}"/>"#).unwrap();
}
// for="5" created_at
if let Some(v) = edge.attributes.get("created_at").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="5" value="{v}"/>"#).unwrap();
}
// for="6" valid_at
if let Some(v) = edge.attributes.get("valid_at").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="6" value="{v}"/>"#).unwrap();
}
// for="7" entity_confidence
if let Some(v) = edge.attributes.get("entity_confidence").and_then(|v| v.as_f64()) {
    writeln!(out, r#"          <attvalue for="7" value="{v}"/>"#).unwrap();
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib handlers::export::tests::test_gexf_emits_new_edge_attributes -- --nocapture 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat(export): add last_activated, created_at, valid_at, entity_confidence to GEXF edges"
```

---

### Task 2: Add new memory node attrs to GEXF emission

Memory nodes already emit `importance`, `tier`, `access_count`, `activation`, `experience_type`. Add `last_accessed`, `temporal_relevance`, `created_at`, `agent_id`, `run_id`.

**Files:**
- Modify: `src/handlers/export.rs:287-296` (node attribute declarations)
- Modify: `src/handlers/export.rs:313-351` (node attvalue emission loop)
- Test: `src/handlers/export.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_gexf_emits_new_memory_attributes() {
    use crate::memory::{Experience, ExperienceType, Memory, MemoryId, MemoryTier};

    let memory = Memory {
        id: MemoryId(Uuid::new_v4()),
        experience: Experience {
            content: "test".into(),
            experience_type: ExperienceType::Observation,
            embeddings: None,
        },
        tier: MemoryTier::Longterm,
        importance_raw: 0.5,
        access_count_raw: 7,
        created_at: Utc::now(),
        last_accessed_at: Utc::now(),
        activation_raw: 0.3,
        entity_refs: vec![],
        agent_id: Some("agent-1".into()),
        run_id: Some("run-42".into()),
    };

    let node = memory_to_node(&memory, false);
    let response = GraphExportResponse {
        metadata: ExportMetadata {
            exported_at: Utc::now(),
            user_id: "u".into(),
            node_count: 1,
            edge_count: 0,
            node_counts_by_type: HashMap::new(),
            edge_counts_by_type: HashMap::new(),
        },
        nodes: vec![node],
        edges: vec![],
    };
    let gexf = to_gexf(&response);

    assert!(gexf.contains(r#"title="last_accessed""#));
    assert!(gexf.contains(r#"title="temporal_relevance""#));
    assert!(gexf.contains(r#"title="created_at""#));
    assert!(gexf.contains(r#"title="agent_id""#));
    assert!(gexf.contains(r#"title="run_id""#));
    assert!(gexf.contains(r#"value="agent-1""#));
    assert!(gexf.contains(r#"value="run-42""#));
}
```

Note: the Memory struct's exact field names may differ — before pasting, open `src/memory.rs` and verify the constructor. If the test helper pattern in the existing module (like `make_entity()`) matches, prefer that.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::export::tests::test_gexf_emits_new_memory_attributes -- --nocapture 2>&1 | tail -20`
Expected: FAIL — assertion on `title="last_accessed"` fails.

- [ ] **Step 3: Expand node attribute declarations**

At `src/handlers/export.rs:287-296`, extend with new IDs 8–15 (preserving existing 0–7):

```rust
writeln!(out, r#"    <attributes class="node">"#).unwrap();
writeln!(out, r#"      <attribute id="0" title="type" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="1" title="importance" type="float"/>"#).unwrap();
writeln!(out, r#"      <attribute id="2" title="salience" type="float"/>"#).unwrap();
writeln!(out, r#"      <attribute id="3" title="tier" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="4" title="access_count" type="integer"/>"#).unwrap();
writeln!(out, r#"      <attribute id="5" title="activation" type="float"/>"#).unwrap();
writeln!(out, r#"      <attribute id="6" title="mention_count" type="integer"/>"#).unwrap();
writeln!(out, r#"      <attribute id="7" title="experience_type" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="8" title="last_accessed" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="9" title="temporal_relevance" type="float"/>"#).unwrap();
writeln!(out, r#"      <attribute id="10" title="created_at" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="11" title="agent_id" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="12" title="run_id" type="string"/>"#).unwrap();
writeln!(out, r#"    </attributes>"#).unwrap();
```

(IDs 13–15 reserved for Task 3's entity attrs; shared node attribute table.)

- [ ] **Step 4: Emit the new attvalues**

After the existing `experience_type` block (around line 351 inside node loop), add:

```rust
// for="8" last_accessed
if let Some(v) = node.attributes.get("last_accessed").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="8" value="{v}"/>"#).unwrap();
}
// for="9" temporal_relevance
if let Some(v) = node.attributes.get("temporal_relevance").and_then(|v| v.as_f64()) {
    writeln!(out, r#"          <attvalue for="9" value="{v}"/>"#).unwrap();
}
// for="10" created_at
if let Some(v) = node.attributes.get("created_at").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="10" value="{v}"/>"#).unwrap();
}
// for="11" agent_id
if let Some(v) = node.attributes.get("agent_id").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="11" value="{v}"/>"#).unwrap();
}
// for="12" run_id
if let Some(v) = node.attributes.get("run_id").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="12" value="{v}"/>"#).unwrap();
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib handlers::export::tests::test_gexf_emits_new_memory_attributes -- --nocapture 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat(export): add last_accessed, temporal_relevance, agent/run ids to GEXF memory nodes"
```

---

### Task 3: Add new entity node attrs to GEXF emission

Entity nodes already emit via `entity_to_node`, which puts `last_seen_at`, `created_at`, `summary`, `labels`, `is_proper_noun` into the JSON attributes. The GEXF emission loop ignores them. Plumb them through.

**Files:**
- Modify: `src/handlers/export.rs:287-296` (node attribute declarations — add IDs 13–17)
- Modify: `src/handlers/export.rs:313-351` (node attvalue loop)
- Test: `src/handlers/export.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_gexf_emits_new_entity_attributes() {
    let mut entity = make_entity();
    entity.labels = vec![EntityLabel::Person, EntityLabel::Concept];
    let node = entity_to_node(&entity, false);
    let response = GraphExportResponse {
        metadata: ExportMetadata {
            exported_at: Utc::now(),
            user_id: "u".into(),
            node_count: 1,
            edge_count: 0,
            node_counts_by_type: HashMap::new(),
            edge_counts_by_type: HashMap::new(),
        },
        nodes: vec![node],
        edges: vec![],
    };
    let gexf = to_gexf(&response);

    assert!(gexf.contains(r#"title="last_seen_at""#));
    assert!(gexf.contains(r#"title="entity_created_at""#));
    assert!(gexf.contains(r#"title="summary""#));
    assert!(gexf.contains(r#"title="labels""#));
    assert!(gexf.contains(r#"title="is_proper_noun""#));
    assert!(gexf.contains(r#"value="The Rust mascot""#));
    assert!(gexf.contains(r#"value="Person,Concept""#));
    assert!(gexf.contains(r#"value="true""#));
}
```

Note: `entity_created_at` disambiguates from memory's `created_at` (ID 10). GEXF requires unique attribute titles per class; since they share the same node attribute table, we prefix.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::export::tests::test_gexf_emits_new_entity_attributes -- --nocapture 2>&1 | tail -20`
Expected: FAIL.

- [ ] **Step 3: Expand node attribute declarations**

Extend the block added in Task 2 with IDs 13–17:

```rust
writeln!(out, r#"      <attribute id="13" title="last_seen_at" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="14" title="entity_created_at" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="15" title="summary" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="16" title="labels" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="17" title="is_proper_noun" type="string"/>"#).unwrap();
```

Place these inside the closing `</attributes>` block for nodes. `is_proper_noun` is typed `string` rather than `boolean` because GEXF 1.3's type system treats booleans inconsistently across parsers; string `"true"`/`"false"` is universally safe.

- [ ] **Step 4: Update `entity_to_node` label serialization**

In `src/handlers/export.rs:80-84`, the current code builds `labels_vec: Vec<String>`. Keep that but also emit a comma-joined string for GEXF (arrays are awkward in GEXF attvalues):

```rust
let labels_vec: Vec<String> = entity
    .labels
    .iter()
    .map(|l| l.as_str().to_owned())
    .collect();
let labels_joined = labels_vec.join(",");

let mut attrs = serde_json::json!({
    "salience": entity.salience,
    "mention_count": entity.mention_count,
    "is_proper_noun": entity.is_proper_noun,
    "labels": labels_vec,
    "labels_joined": labels_joined,
    "created_at": entity.created_at,
    "last_seen_at": entity.last_seen_at,
    "summary": entity.summary,
});
```

- [ ] **Step 5: Emit the new attvalues**

After the `run_id` emission added in Task 2, add:

```rust
// for="13" last_seen_at
if let Some(v) = node.attributes.get("last_seen_at").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="13" value="{v}"/>"#).unwrap();
}
// for="14" entity_created_at (only for entity nodes — reuses created_at JSON key;
// emit only when node_type == "entity" to avoid double-emission w/ memory's id=10).
if node.node_type == "entity" {
    if let Some(v) = node.attributes.get("created_at").and_then(|v| v.as_str()) {
        let v = xml_escape(v);
        writeln!(out, r#"          <attvalue for="14" value="{v}"/>"#).unwrap();
    }
}
// for="15" summary
if let Some(v) = node.attributes.get("summary").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="15" value="{v}"/>"#).unwrap();
}
// for="16" labels
if let Some(v) = node.attributes.get("labels_joined").and_then(|v| v.as_str()) {
    let v = xml_escape(v);
    writeln!(out, r#"          <attvalue for="16" value="{v}"/>"#).unwrap();
}
// for="17" is_proper_noun
if let Some(v) = node.attributes.get("is_proper_noun").and_then(|v| v.as_bool()) {
    writeln!(out, r#"          <attvalue for="17" value="{v}"/>"#).unwrap();
}
```

Guard `for="10" created_at` (added in Task 2) similarly so it only emits for `node_type == "memory"`:

```rust
// for="10" created_at (memory nodes only — entity uses id=14)
if node.node_type == "memory" {
    if let Some(v) = node.attributes.get("created_at").and_then(|v| v.as_str()) {
        let v = xml_escape(v);
        writeln!(out, r#"          <attvalue for="10" value="{v}"/>"#).unwrap();
    }
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test --lib handlers::export::tests -- --nocapture 2>&1 | tail -20`
Expected: PASS on `test_gexf_emits_new_entity_attributes` AND all prior GEXF tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat(export): add last_seen_at, summary, labels, is_proper_noun to GEXF entity nodes"
```

---

### Task 4: Add episode node attrs + per-type created_at disambiguation

Episode nodes already emit `type` via `for="0"`. Add `source`, `valid_at`, `episode_created_at` (disambiguated from memory's `created_at`).

**Files:**
- Modify: `src/handlers/export.rs` (node attributes declarations + emission)
- Test: `src/handlers/export.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_gexf_emits_new_episode_attributes() {
    let episode = EpisodicNode {
        uuid: Uuid::new_v4(),
        name: "Meeting".into(),
        content: "...".into(),
        valid_at: Utc::now(),
        created_at: Utc::now(),
        entity_refs: vec![],
        source: EpisodeSource::Message,
        metadata: HashMap::new(),
        extracted_triples: vec![],
    };
    let node = episode_to_node(&episode);
    let response = GraphExportResponse {
        metadata: ExportMetadata {
            exported_at: Utc::now(),
            user_id: "u".into(),
            node_count: 1,
            edge_count: 0,
            node_counts_by_type: HashMap::new(),
            edge_counts_by_type: HashMap::new(),
        },
        nodes: vec![node],
        edges: vec![],
    };
    let gexf = to_gexf(&response);

    assert!(gexf.contains(r#"title="source""#));
    assert!(gexf.contains(r#"title="valid_at""#));
    assert!(gexf.contains(r#"title="episode_created_at""#));
    assert!(gexf.contains(r#"value="Message""#));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::export::tests::test_gexf_emits_new_episode_attributes 2>&1 | tail -10`
Expected: FAIL.

- [ ] **Step 3: Expand node attribute declarations (IDs 18–20)**

```rust
writeln!(out, r#"      <attribute id="18" title="source" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="19" title="valid_at" type="string"/>"#).unwrap();
writeln!(out, r#"      <attribute id="20" title="episode_created_at" type="string"/>"#).unwrap();
```

- [ ] **Step 4: Emit the new attvalues**

Inside the node loop, guarded on `node_type == "episode"`:

```rust
if node.node_type == "episode" {
    if let Some(v) = node.attributes.get("source").and_then(|v| v.as_str()) {
        let v = xml_escape(v);
        writeln!(out, r#"          <attvalue for="18" value="{v}"/>"#).unwrap();
    }
    if let Some(v) = node.attributes.get("valid_at").and_then(|v| v.as_str()) {
        let v = xml_escape(v);
        writeln!(out, r#"          <attvalue for="19" value="{v}"/>"#).unwrap();
    }
    if let Some(v) = node.attributes.get("created_at").and_then(|v| v.as_str()) {
        let v = xml_escape(v);
        writeln!(out, r#"          <attvalue for="20" value="{v}"/>"#).unwrap();
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib handlers::export::tests -- --nocapture 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat(export): add source, valid_at, episode_created_at to GEXF episode nodes"
```

---

### Task 5: Add `server_time` to GEXF `<meta>` block

Per spec, `<meta>` gains a `server_time` RFC-3339 field so the client can detect clock skew and expose it in the sidebar.

**Files:**
- Modify: `src/handlers/export.rs:277-279` (meta block)
- Test: `src/handlers/export.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_gexf_meta_includes_server_time() {
    let response = GraphExportResponse {
        metadata: ExportMetadata {
            exported_at: Utc::now(),
            user_id: "u".into(),
            node_count: 0,
            edge_count: 0,
            node_counts_by_type: HashMap::new(),
            edge_counts_by_type: HashMap::new(),
        },
        nodes: vec![],
        edges: vec![],
    };
    let gexf = to_gexf(&response);

    assert!(gexf.contains("<server_time>"));
    assert!(gexf.contains("</server_time>"));
    // The value is whatever exported_at serializes to (RFC-3339)
    let expected = response.metadata.exported_at.to_rfc3339();
    assert!(
        gexf.contains(&expected),
        "expected {expected} in GEXF meta"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::export::tests::test_gexf_meta_includes_server_time 2>&1 | tail -10`
Expected: FAIL.

- [ ] **Step 3: Implement**

Replace the meta block at `src/handlers/export.rs:277-279`:

```rust
let server_time = export.metadata.exported_at.to_rfc3339();
writeln!(out, r#"  <meta lastmodifieddate="{date}">"#).unwrap();
writeln!(out, r#"    <creator>shodh-memory</creator>"#).unwrap();
writeln!(out, r#"    <server_time>{server_time}</server_time>"#).unwrap();
writeln!(out, r#"  </meta>"#).unwrap();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib handlers::export::tests::test_gexf_meta_includes_server_time 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat(export): emit server_time in GEXF <meta> for clock-skew detection"
```

---

### Task 6: Gate large content fields behind `?include_content=bool`

Mirror the existing `?include_embeddings=bool` pattern. When `false` (default), omit `content` from memory nodes, `summary` from entity nodes, `content` from episode nodes. The viewer lazy-fetches full content on node click.

**Files:**
- Modify: `src/handlers/export.rs:420-428` (`ExportParams` struct)
- Modify: `src/handlers/export.rs:76-114` (`entity_to_node`)
- Modify: `src/handlers/export.rs:120-164` (`memory_to_node`)
- Modify: `src/handlers/export.rs:167-181` (`episode_to_node`)
- Modify: `src/handlers/export.rs:449-552` (`export_graph` handler — pass flag through)
- Test: `src/handlers/export.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_memory_to_node_omits_content_when_flag_false() {
    use crate::memory::{Experience, ExperienceType, Memory, MemoryId, MemoryTier};
    let memory = Memory {
        id: MemoryId(Uuid::new_v4()),
        experience: Experience {
            content: "secret sauce".into(),
            experience_type: ExperienceType::Observation,
            embeddings: None,
        },
        tier: MemoryTier::Working,
        importance_raw: 0.5,
        access_count_raw: 1,
        created_at: Utc::now(),
        last_accessed_at: Utc::now(),
        activation_raw: 0.0,
        entity_refs: vec![],
        agent_id: None,
        run_id: None,
    };

    let with = memory_to_node(&memory, false, true);
    assert!(with.attributes.get("content").is_some());

    let without = memory_to_node(&memory, false, false);
    assert!(without.attributes.get("content").is_none());
    // Label still truncated from content — that's fine, it's short.
    assert!(!without.label.is_empty());
}
```

(Signature change: `memory_to_node(memory, include_embeddings, include_content)`. Same for `entity_to_node` gaining a third `include_content` bool governing `summary`. `episode_to_node` gains `include_content` governing `content`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::export::tests::test_memory_to_node_omits_content_when_flag_false 2>&1 | tail -10`
Expected: FAIL — wrong number of arguments.

- [ ] **Step 3: Update `ExportParams`**

At `src/handlers/export.rs:420-428`:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct ExportParams {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_include")]
    pub include: String,
    #[serde(default)]
    pub min_importance: f32,
    #[serde(default)]
    pub include_embeddings: bool,
    #[serde(default)]
    pub include_content: bool,
}
```

- [ ] **Step 4: Update node builders to accept the flag**

`memory_to_node`:
```rust
pub fn memory_to_node(
    memory: &crate::memory::Memory,
    include_embeddings: bool,
    include_content: bool,
) -> ExportNode {
    let label = if memory.experience.content.len() > 100 {
        let boundary = memory.experience.content.floor_char_boundary(97);
        format!("{}...", &memory.experience.content[..boundary])
    } else {
        memory.experience.content.clone()
    };

    let mut attrs = serde_json::json!({
        "importance": memory.importance(),
        "tier": format!("{:?}", memory.tier),
        "access_count": memory.access_count(),
        "last_accessed": memory.last_accessed(),
        "temporal_relevance": memory.temporal_relevance(),
        "activation": memory.activation(),
        "experience_type": format!("{:?}", memory.experience.experience_type),
        "created_at": memory.created_at,
    });

    if include_content {
        attrs["content"] = serde_json::Value::String(memory.experience.content.clone());
    }
    // ... rest unchanged (agent_id, run_id, embeddings)
}
```

`entity_to_node`: same pattern — move `"summary": entity.summary` out of the initial JSON macro and gate it:

```rust
if include_content {
    attrs["summary"] = serde_json::Value::String(entity.summary.clone());
}
```

`episode_to_node`: add `include_content` parameter, gate the `content` key the same way.

- [ ] **Step 5: Update call sites in `export_graph`**

`src/handlers/export.rs:469`:
```rust
nodes.push(entity_to_node(entity, params.include_embeddings, params.include_content));
```

`src/handlers/export.rs:481`:
```rust
nodes.push(episode_to_node(episode, params.include_content));
```

`src/handlers/export.rs:510`:
```rust
nodes.push(memory_to_node(memory, params.include_embeddings, params.include_content));
```

Update all pre-existing test call sites too (grep for `memory_to_node(`, `entity_to_node(`, `episode_to_node(`).

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test --lib handlers::export::tests 2>&1 | tail -30`
Expected: PASS on all tests (both new and existing).

- [ ] **Step 7: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat(export): gate large content fields behind ?include_content=bool"
```

---

### Task 7: Add ETag / If-None-Match to export handler

Emit weak ETag derived from the metadata's `exported_at` timestamp (simple and correct: any write changes the server-side max `last_modified`, which affects `exported_at` only insofar as it reflects the snapshot). For v1, use a hash of the serialized response — cheap, correct, no cache-invalidation concerns.

**Files:**
- Modify: `src/handlers/export.rs` (handler response building)
- Test: `src/handlers/export.rs` (handler integration test)

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn test_export_handler_emits_etag() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    let (app, _user_id) = crate::handlers::test_helpers::app_with_seed_data().await;

    let req = Request::builder()
        .uri("/api/graph/test_user/export?format=gexf")
        .header("X-API-Key", "test-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let etag = resp
        .headers()
        .get("etag")
        .expect("etag header present")
        .to_str()
        .unwrap()
        .to_owned();
    assert!(etag.starts_with("W/\""), "weak etag expected: {etag}");

    // Second request with If-None-Match should return 304
    let req2 = Request::builder()
        .uri("/api/graph/test_user/export?format=gexf")
        .header("X-API-Key", "test-key")
        .header("If-None-Match", &etag)
        .body(Body::empty())
        .unwrap();
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::NOT_MODIFIED);
}
```

(If `app_with_seed_data` doesn't exist in `test_helpers`, write the simpler variant that constructs a `MultiUserMemoryManager` + nests into a `Router` with auth middleware + seeds a handful of entities.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::export::tests::test_export_handler_emits_etag 2>&1 | tail -20`
Expected: FAIL — no etag header.

- [ ] **Step 3: Implement ETag emission + If-None-Match handling**

In `export_graph` at `src/handlers/export.rs:449-552`, replace the final match:

```rust
use axum::http::{header, HeaderMap, StatusCode};

// --- Build payload ---
let body = match params.format.as_str() {
    "gexf" => to_gexf(&response),
    _ => serde_json::to_string(&response).map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?,
};

// Weak ETag: hash of body (cheap, correct). Blake3 already in deps for embeddings.
let hash = blake3::hash(body.as_bytes()).to_hex();
let etag = format!(r#"W/"{}""#, &hash[..16]);

// If-None-Match → 304
if let Some(inm) = request_headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()) {
    if inm == etag {
        let mut headers = HeaderMap::new();
        headers.insert(header::ETAG, etag.parse().unwrap());
        return Ok((StatusCode::NOT_MODIFIED, headers).into_response());
    }
}

let content_type = if params.format == "gexf" {
    "application/gexf+xml"
} else {
    "application/json"
};

let mut headers = HeaderMap::new();
headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
headers.insert(header::ETAG, etag.parse().unwrap());
Ok((headers, body).into_response())
```

The handler needs to receive `HeaderMap` for request_headers. Add to the signature:

```rust
pub async fn export_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(params): Query<ExportParams>,
    request_headers: HeaderMap,
) -> Result<axum::response::Response, AppError>
```

Axum extracts `HeaderMap` automatically when present in the handler arg list.

- [ ] **Step 4: Verify blake3 is a direct dep**

Run: `grep -F 'blake3' Cargo.toml`
Expected: a line like `blake3 = "1.x"`. If missing, use a different hasher — `sha2` is almost certainly present. If neither, fall back to `std::hash::DefaultHasher` for a shorter but still collision-acceptable ETag.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib handlers::export::tests::test_export_handler_emits_etag 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 6: Clippy clean**

Run: `cargo clippy --lib -- -D warnings 2>&1 | tail -20`
Expected: no warnings in modified code.

- [ ] **Step 7: Commit**

```bash
git add src/handlers/export.rs Cargo.toml Cargo.lock
git commit -m "feat(export): weak ETag + If-None-Match for unchanged-snapshot short-circuit"
```

---

## Phase 2 — SSE Query-Param Auth Fallback

### Task 8: Allow `?api_key=X` query param auth on `/api/events/sse`

The auth middleware at `src/auth.rs:200-260` already accepts `?api_key=` **only** for WebSocket upgrades. Extend the allow-list to also permit it for `/api/events/sse`.

**Files:**
- Modify: `src/auth.rs:229-247` (query-param auth branch)
- Test: `src/auth.rs` (existing test module)

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/auth.rs`:

```rust
#[tokio::test]
async fn auth_accepts_query_param_for_sse() {
    let _lock = ENV_LOCK.lock().unwrap();
    clear_auth_env();
    env::set_var("SHODH_DEV_API_KEY", "sse-key");

    // Build a minimal router with auth middleware
    let app = Router::new()
        .route("/api/events/sse", axum::routing::get(|| async { "ok" }))
        .layer(axum::middleware::from_fn(auth_middleware));

    let req = Request::builder()
        .method("GET")
        .uri("/api/events/sse?user_id=u&api_key=sse-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_rejects_query_param_for_non_sse_non_ws() {
    let _lock = ENV_LOCK.lock().unwrap();
    clear_auth_env();
    env::set_var("SHODH_DEV_API_KEY", "the-key");

    let app = Router::new()
        .route("/api/memories", axum::routing::get(|| async { "ok" }))
        .layer(axum::middleware::from_fn(auth_middleware));

    let req = Request::builder()
        .method("GET")
        .uri("/api/memories?api_key=the-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib auth::tests::auth_accepts_query_param_for_sse auth::tests::auth_rejects_query_param_for_non_sse_non_ws 2>&1 | tail -20`
Expected: FAIL on `auth_accepts_query_param_for_sse` (SSE path isn't whitelisted).

- [ ] **Step 3: Extend the allow-list**

At `src/auth.rs:229-247`, replace the `.or_else` block that currently checks only WebSocket:

```rust
.or_else(|| {
    // Browser-compatibility fallback: allow `?api_key=...` for endpoints
    // that cannot set custom headers (WebSocket upgrades, SSE EventSource).
    // This leaks the key into URLs/logs, so it's explicitly allow-listed
    // rather than available everywhere.
    let is_websocket = request
        .headers()
        .get("upgrade")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);
    let is_sse_path = path == "/api/events/sse";
    if !is_websocket && !is_sse_path {
        return None;
    }
    request.uri().query().and_then(|q| {
        q.split('&')
            .find_map(|pair| pair.strip_prefix("api_key=").map(|v| v.to_string()))
    })
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib auth::tests 2>&1 | tail -30`
Expected: PASS on all auth tests — both new tests AND all existing ones.

- [ ] **Step 5: Clippy clean**

Run: `cargo clippy --lib -- -D warnings 2>&1 | tail -10`

- [ ] **Step 6: Commit**

```bash
git add src/auth.rs
git commit -m "feat(auth): allow ?api_key= query-param fallback for /api/events/sse

Browser EventSource cannot set X-API-Key; existing allow-list already
permits query-param auth for WebSocket upgrades. Extend it to cover
the SSE endpoint so the new viewer can subscribe to live updates."
```

---

## Phase 3 — Vendor the Viewer Fork

### Task 9: Vendor `../viewer/` fork at `827c191` into `src/handlers/viewer/`

Per spec Implementation Prerequisites: this is step 0. Commit creates provenance; every subsequent viewer change modifies this vendored base.

**Files:**
- Create: `src/handlers/viewer/` (entire subtree, populated from fork)

- [ ] **Step 1: Verify `../viewer/` exists and the commit is reachable**

Run: `git -C ../viewer rev-parse 827c191 && git -C ../viewer log --oneline -3 827c191`
Expected: prints commit hash + 3 most recent commits at that ref.
If the ref isn't reachable, fall back to `75327d2` as noted in the spec's Fork Base section.

- [ ] **Step 2: Clone the fork to a temp worktree**

Run:
```bash
TMPDIR=$(mktemp -d)
git -C ../viewer worktree add --detach "$TMPDIR/viewer-fork" 827c191
ls "$TMPDIR/viewer-fork"
```
Expected: directory listing shows the viewer source tree (index.html, js/, css/, etc.).

- [ ] **Step 3: Strip HTTP-Graph and ISAP domain modules**

From `$TMPDIR/viewer-fork`, remove:
- Any `js/**/http-graph*`, `js/**/isap*`, or equivalently-named modules
- Any CSS specifically for the HTTP-Graph or ISAP sidebars
- Any fixture / test data referencing HTTP redirect chains or ISAP campaigns

Identify them with:
```bash
grep -rl 'http-graph\|isap\|ISAP\|redirect.chain' "$TMPDIR/viewer-fork" | sort -u
```

Inspect each match; delete the ones that are domain-specific. Keep modules that are generic (sigma bootstrap, GEXF loader, FA2 worker, selection, hop-highlight, sidebar scaffold, detail-panel scaffold, drag-drop).

- [ ] **Step 4: Copy the stripped tree into the repo**

```bash
mkdir -p src/handlers/viewer
cp -R "$TMPDIR/viewer-fork/." src/handlers/viewer/
# Drop any .git / node_modules that tagged along
rm -rf src/handlers/viewer/.git src/handlers/viewer/node_modules
```

Verify with `ls src/handlers/viewer/`.

- [ ] **Step 5: Ensure the vendored tree does NOT include vendored JS deps yet**

Task 10 handles those. If the fork already contained e.g. `vendor/sigma.min.js`, move them to `src/handlers/assets/` now to centralize. If they're CDN-imported via importmap in the fork, leave the importmap; we'll rewrite the origins in Task 10.

Run: `find src/handlers/viewer -name '*.js' -size +100k`
Expected: either no results (CDN-imported) or a small handful (vendored). Note the list for Task 10.

- [ ] **Step 6: Clean up worktree**

```bash
git -C ../viewer worktree remove "$TMPDIR/viewer-fork"
rm -rf "$TMPDIR"
```

- [ ] **Step 7: Commit as the provenance commit**

```bash
git add src/handlers/viewer
git commit -m "feat(viewer): vendor ../viewer/ fork @ 827c191 into src/handlers/viewer/

Stripped HTTP-Graph and ISAP domain modules; kept the generic
graphology / sigma / GEXF / FA2 / interaction scaffolding. All
subsequent viewer work modifies this vendored base.

Source: ../viewer/@827c191
(refactor(viewer): simplify dashed edge program and tune redirect edges)"
```

---

### Task 10: Vendor sigma.js / graphology / graphology-gexf / FA2 as assets

Mirror the `d3.v7.9.0.min.js` pattern: pull the library bundles into `src/handlers/assets/`, extend the `graph_asset` allow-list.

**Files:**
- Create: `src/handlers/assets/sigma.min.js`
- Create: `src/handlers/assets/graphology.umd.min.js`
- Create: `src/handlers/assets/graphology-gexf.umd.min.js`
- Create: `src/handlers/assets/graphology-layout-forceatlas2.umd.min.js`
- Modify: `src/handlers/visualization.rs:267-283` (`graph_asset` handler)

- [ ] **Step 1: Download library bundles**

Pin versions that match what the vendored fork used. Check the fork's existing HTML or `package.json` (if any) for the versions it imported:

```bash
grep -rE 'sigma|graphology' src/handlers/viewer/*.html src/handlers/viewer/**/*.json 2>/dev/null | head
```

Then fetch the UMD bundles (example using a recent known-good set — adjust to match):

```bash
# Sigma (v3.0.1 as of writing — confirm from fork)
curl -L https://cdn.jsdelivr.net/npm/sigma@3.0.1/dist/sigma.min.js \
  -o src/handlers/assets/sigma.min.js
# graphology
curl -L https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js \
  -o src/handlers/assets/graphology.umd.min.js
# graphology-gexf
curl -L https://cdn.jsdelivr.net/npm/graphology-gexf@0.10.2/dist/graphology-gexf.umd.min.js \
  -o src/handlers/assets/graphology-gexf.umd.min.js
# graphology-layout-forceatlas2
curl -L https://cdn.jsdelivr.net/npm/graphology-layout-forceatlas2@0.10.1/dist/graphology-layout-forceatlas2.umd.min.js \
  -o src/handlers/assets/graphology-layout-forceatlas2.umd.min.js
```

Verify each:
```bash
ls -la src/handlers/assets/*.min.js
```
Expected: all four files, sizes between 50KB and 500KB. If a URL 404s, look up the current latest on npmjs.com and substitute.

- [ ] **Step 2: Extend the `graph_asset` allow-list**

In `src/handlers/visualization.rs:267-283`, add the new entries to the match:

```rust
pub async fn graph_asset(Path(file): Path<String>) -> Response {
    let bytes: &'static [u8] = match file.as_str() {
        "d3.v7.9.0.min.js" => include_bytes!("assets/d3.v7.9.0.min.js"),
        "three.module.js" => include_bytes!("assets/three.module.js"),
        "three.core.js" => include_bytes!("assets/three.core.js"),
        "OrbitControls.js" => include_bytes!("assets/OrbitControls.js"),
        "sigma.min.js" => include_bytes!("assets/sigma.min.js"),
        "graphology.umd.min.js" => include_bytes!("assets/graphology.umd.min.js"),
        "graphology-gexf.umd.min.js" => include_bytes!("assets/graphology-gexf.umd.min.js"),
        "graphology-layout-forceatlas2.umd.min.js" => {
            include_bytes!("assets/graphology-layout-forceatlas2.umd.min.js")
        }
        _ => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        ],
        bytes,
    )
        .into_response()
}
```

- [ ] **Step 3: Write an integration test for the allow-list**

Add to `src/handlers/visualization.rs` (or sibling test file if patterns differ):

```rust
#[tokio::test]
async fn graph_asset_serves_vendored_sigma() {
    use axum::{body::Body, http::Request, Router};
    use tower::ServiceExt;

    let app = Router::new().route(
        "/graph/assets/{file}",
        axum::routing::get(graph_asset),
    );

    let req = Request::builder()
        .uri("/graph/assets/sigma.min.js")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "application/javascript; charset=utf-8"
    );
}

#[tokio::test]
async fn graph_asset_rejects_unknown_file() {
    use axum::{body::Body, http::Request, Router};
    use tower::ServiceExt;

    let app = Router::new().route(
        "/graph/assets/{file}",
        axum::routing::get(graph_asset),
    );

    let req = Request::builder()
        .uri("/graph/assets/../Cargo.toml")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib handlers::visualization::graph_asset 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/handlers/assets/ src/handlers/visualization.rs
git commit -m "feat(viewer): vendor sigma, graphology, graphology-gexf, FA2 as static assets"
```

---

## Phase 4 — Rust Handler Glue

### Task 11: Add `GET /graph/view2` handler with nonce + API_KEY substitution

New handler; follows the existing `graph_view` pattern at `src/handlers/visualization.rs:257-264`.

**Files:**
- Modify: `src/handlers/visualization.rs` (add `graph_view2` handler + helper)
- Modify: `src/handlers/router.rs` (register the route in `build_public_routes`)
- Test: `src/handlers/visualization.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn graph_view2_responds_with_html_and_substitutes_placeholders() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    std::env::set_var("SHODH_DEV_API_KEY", "view2-key");
    let app = axum::Router::new().route(
        "/graph/view2",
        axum::routing::get(graph_view2),
    );

    let req = Request::builder()
        .uri("/graph/view2?user_id=alice")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body_bytes.to_vec()).unwrap();
    assert!(body.contains("<!DOCTYPE html>"));
    assert!(body.contains("view2-key"), "API key not substituted");
    assert!(!body.contains("{{API_KEY}}"), "template placeholder leaked");
    assert!(!body.contains("{{NONCE}}"), "nonce placeholder leaked");
    std::env::remove_var("SHODH_DEV_API_KEY");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::visualization::graph_view2_responds_with_html 2>&1 | tail -10`
Expected: FAIL — `graph_view2` not defined.

- [ ] **Step 3: Implement the handler**

Add to `src/handlers/visualization.rs` near the existing `graph_view`:

```rust
const VIEWER_HTML: &str = include_str!("viewer/index.html");

/// GET /graph/view2 - Serve the sigma.js GEXF viewer
pub async fn graph_view2(Query(params): Query<GraphViewParams>) -> Response {
    let user_id = params.user_id.unwrap_or_else(|| "default".to_string());
    let nonce = generate_nonce();
    let api_key = std::env::var("SHODH_DEV_API_KEY").unwrap_or_default();

    let body = VIEWER_HTML
        .replace("{{NONCE}}", &nonce)
        .replace("{{API_KEY}}", &api_key)
        .replace("{{USER_ID}}", &user_id);

    let mut response = Html(body).into_response();
    response.extensions_mut().insert(CspNonce(nonce));
    response
}
```

- [ ] **Step 4: Ensure `src/handlers/viewer/index.html` carries the placeholders**

Viewer HTML (from the vendored fork) needs the `{{NONCE}}`, `{{API_KEY}}`, `{{USER_ID}}` placeholders. In the `<script>` and `<link>` tags that load local assets, add `nonce="{{NONCE}}"`. Add near the top of `<head>`:

```html
<script nonce="{{NONCE}}">
  window.SHODH_API_KEY = "{{API_KEY}}";
  window.SHODH_USER_ID = "{{USER_ID}}";
</script>
```

Confirm with: `grep -F '{{' src/handlers/viewer/index.html`.

- [ ] **Step 5: Run the handler test**

Run: `cargo test --lib handlers::visualization::graph_view2_responds_with_html 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 6: Register the route**

In `src/handlers/router.rs:58-59` (same block as `/graph/view`), add:

```rust
.route("/graph/view2", get(visualization::graph_view2))
```

Run: `cargo check --lib 2>&1 | tail -10` — verify no compile errors.

- [ ] **Step 7: Commit**

```bash
git add src/handlers/visualization.rs src/handlers/router.rs src/handlers/viewer/index.html
git commit -m "feat(viewer): GET /graph/view2 handler with nonce + API_KEY substitution"
```

---

### Task 12: Expand `graph_asset` to serve the viewer's own JS/CSS modules

Viewer modules live in `src/handlers/viewer/js/**/*.js` and `src/handlers/viewer/css/*.css`. Either extend `graph_asset` with every path, or add a sibling `viewer_asset` handler rooted at `/graph/viewer/{*rest}`. Prefer the latter for clarity and to avoid a 30-arm match.

**Files:**
- Create: `src/handlers/viewer_asset.rs`
- Modify: `src/handlers/mod.rs` (register the module)
- Modify: `src/handlers/router.rs` (route `/graph/viewer/{*rest}`)

- [ ] **Step 1: Write the failing test**

```rust
// src/handlers/viewer_asset.rs (tests module at bottom)
#[tokio::test]
async fn viewer_asset_serves_boot_js() {
    use axum::{body::Body, http::Request, Router};
    use tower::ServiceExt;

    let app = Router::new().route(
        "/graph/viewer/{*rest}",
        axum::routing::get(viewer_asset),
    );

    let req = Request::builder()
        .uri("/graph/viewer/js/boot.js")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "application/javascript; charset=utf-8"
    );
}

#[tokio::test]
async fn viewer_asset_rejects_path_traversal() {
    use axum::{body::Body, http::Request, Router};
    use tower::ServiceExt;

    let app = Router::new().route(
        "/graph/viewer/{*rest}",
        axum::routing::get(viewer_asset),
    );

    for bad in ["../Cargo.toml", "js/../../Cargo.toml", "js/../../../etc/passwd"] {
        let req = Request::builder()
            .uri(format!("/graph/viewer/{bad}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND, "bad path {bad}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib handlers::viewer_asset 2>&1 | tail -10`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement via `rust-embed`**

`include_bytes!` can't do dynamic file lookup, so use `rust-embed` (compile-time embedding with runtime lookup). Add to `Cargo.toml`:

```toml
rust-embed = "8"
```

Create `src/handlers/viewer_asset.rs`:

```rust
//! Serve bundled viewer assets (JS modules, CSS) under /graph/viewer/{path}.
//!
//! Files are embedded at compile time via rust-embed. Path validation is
//! enforced at the URL-router level (segments cannot escape the embed root).

use axum::{
    extract::Path,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "src/handlers/viewer/"]
#[exclude = "*.rs"]
struct ViewerAssets;

fn content_type_for(path: &str) -> &'static str {
    if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    }
}

/// GET /graph/viewer/{*rest} — serve viewer bundle assets
pub async fn viewer_asset(Path(rest): Path<String>) -> Response {
    // Reject traversal attempts (rust-embed already normalizes, but be explicit).
    if rest.contains("..") {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let Some(file) = ViewerAssets::get(&rest) else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    (
        [
            (header::CONTENT_TYPE, content_type_for(&rest)),
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        ],
        file.data.into_owned(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    // ...the two tests from Step 1 go here
}
```

- [ ] **Step 4: Register the module and route**

In `src/handlers/mod.rs`:
```rust
pub mod viewer_asset;
```

In `src/handlers/router.rs`, alongside `/graph/assets/{file}`:
```rust
.route("/graph/viewer/{*rest}", get(viewer_asset::viewer_asset))
```

- [ ] **Step 5: Run tests**

Run: `cargo test --lib handlers::viewer_asset 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 6: Clippy clean**

Run: `cargo clippy --lib -- -D warnings 2>&1 | tail -10`

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/handlers/viewer_asset.rs src/handlers/mod.rs src/handlers/router.rs
git commit -m "feat(viewer): /graph/viewer/{*rest} handler via rust-embed for bundled modules"
```

---

## Phase 5 — Domain Styling (Frontend)

The vendored fork already has generic node/edge rendering. Replace its domain-specific style modules with shodh-specific ones.

### Task 13: Node styling — tier color, importance size, activation halo

**Files:**
- Create: `src/handlers/viewer/js/domain/node-style.js`
- Remove (if present from fork): any `js/domain/*-node-style.js` not matching shodh
- Test: `src/handlers/viewer/tests/unit/node-style.test.js`

- [ ] **Step 1: Write the failing test (Vitest)**

Create `src/handlers/viewer/tests/unit/node-style.test.js`:

```javascript
import { describe, it, expect } from 'vitest';
import { nodeReducer } from '../../js/domain/node-style.js';

describe('nodeReducer', () => {
  it('colors working-tier nodes warm and longterm cool', () => {
    const now = Date.now();
    const working = nodeReducer('n1', {
      type: 'memory',
      tier: 'Working',
      importance: 0.5,
      activation: 0.0,
      access_count: 1,
      last_accessed: new Date(now).toISOString(),
    }, { now });
    const longterm = nodeReducer('n2', {
      type: 'memory',
      tier: 'Longterm',
      importance: 0.5,
      activation: 0.0,
      access_count: 1,
      last_accessed: new Date(now).toISOString(),
    }, { now });

    // Warm = higher R channel, cool = higher B channel
    expect(parseInt(working.color.slice(1, 3), 16)).toBeGreaterThan(
      parseInt(longterm.color.slice(1, 3), 16)
    );
  });

  it('maps importance linearly to size in 6–24px', () => {
    const attrs = { type: 'memory', tier: 'Longterm', activation: 0, access_count: 0,
                    last_accessed: new Date().toISOString() };
    const low = nodeReducer('n1', { ...attrs, importance: 0.0 }, { now: Date.now() });
    const high = nodeReducer('n2', { ...attrs, importance: 1.0 }, { now: Date.now() });
    expect(low.size).toBeCloseTo(6, 1);
    expect(high.size).toBeCloseTo(24, 1);
  });

  it('boosts halo when activation > 0.7', () => {
    const attrs = { type: 'memory', tier: 'Session', importance: 0.5, access_count: 1,
                    last_accessed: new Date().toISOString() };
    const low = nodeReducer('n1', { ...attrs, activation: 0.1 }, { now: Date.now() });
    const high = nodeReducer('n2', { ...attrs, activation: 0.8 }, { now: Date.now() });
    expect(high.haloPulse).toBe(true);
    expect(low.haloPulse).toBe(false);
  });

  it('shapes differ for entity (square) and episode (diamond)', () => {
    const e = nodeReducer('e', { type: 'entity' }, { now: Date.now() });
    const ep = nodeReducer('ep', { type: 'episode' }, { now: Date.now() });
    expect(e.shape).toBe('square');
    expect(ep.shape).toBe('diamond');
  });
});
```

- [ ] **Step 2: Install Vitest in the viewer (one-time)**

Viewer has no `package.json` today — the spec says "No build step. No bundler. No package.json." But tests need a runtime. Use a minimal dev-only `package.json` that's gitignored from production:

Create `src/handlers/viewer/package.json`:
```json
{
  "name": "shodh-viewer",
  "private": true,
  "type": "module",
  "devDependencies": {
    "vitest": "^2.0.0",
    "jsdom": "^24.0.0"
  },
  "scripts": {
    "test": "vitest run"
  }
}
```

Run: `cd src/handlers/viewer && npm install && cd -`

Add `src/handlers/viewer/node_modules/` to `.gitignore` at repo root.

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/node-style.test.js 2>&1 | tail -20 && cd -
```
Expected: FAIL — `node-style.js` doesn't export `nodeReducer`.

- [ ] **Step 4: Implement**

`src/handlers/viewer/js/domain/node-style.js`:

```javascript
// Color tables. Warm → cool = hot → established.
const TIER_COLOR = {
  Working:  '#ff6b2c',
  Session:  '#f5b73b',
  Longterm: '#4b8bb5',
};
const ENTITY_COLOR = '#7d74c9';
const EPISODE_COLOR = '#39a887';
const DEFAULT_COLOR = '#888';

/**
 * Map a graph node's attributes to a sigma reducer result.
 * Pure function; call it from the sigma reducer with the current clock.
 *
 * @param {string} id
 * @param {object} attrs  graphology node attributes
 * @param {{now:number}} ctx  ctx.now is ms since epoch
 * @returns {object} sigma node spec: {size,color,shape,...,haloPulse}
 */
export function nodeReducer(id, attrs, ctx) {
  const type = attrs.type || 'memory';
  const importance = typeof attrs.importance === 'number' ? attrs.importance : 0.5;
  const activation = typeof attrs.activation === 'number' ? attrs.activation : 0.0;

  const size = 6 + 18 * Math.min(1, Math.max(0, importance));
  let color = DEFAULT_COLOR;
  let shape = 'circle';
  if (type === 'memory') {
    color = TIER_COLOR[attrs.tier] || DEFAULT_COLOR;
  } else if (type === 'entity') {
    color = ENTITY_COLOR;
    shape = 'square';
  } else if (type === 'episode') {
    color = EPISODE_COLOR;
    shape = 'diamond';
  }

  const haloPulse = activation > 0.7;

  // Recency badge: last_accessed within 60s
  let recencyBadge = false;
  if (attrs.last_accessed) {
    const ts = Date.parse(attrs.last_accessed);
    if (!Number.isNaN(ts) && (ctx.now - ts) < 60_000) recencyBadge = true;
  }

  return { size, color, shape, haloPulse, recencyBadge, type, label: attrs.label || id };
}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/node-style.test.js 2>&1 | tail -20 && cd -
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/handlers/viewer/js/domain/node-style.js src/handlers/viewer/tests/unit/node-style.test.js src/handlers/viewer/package.json src/handlers/viewer/package-lock.json .gitignore
git commit -m "feat(viewer): tier-keyed node styling (warm/cool scale, activation halo, recency badge)"
```

---

### Task 14: Edge styling — weight thickness, LTP dash, tier hue, activation pulse

**Files:**
- Create: `src/handlers/viewer/js/domain/edge-style.js`
- Test: `src/handlers/viewer/tests/unit/edge-style.test.js`

- [ ] **Step 1: Write the failing test**

`src/handlers/viewer/tests/unit/edge-style.test.js`:

```javascript
import { describe, it, expect } from 'vitest';
import { edgeReducer } from '../../js/domain/edge-style.js';

describe('edgeReducer', () => {
  it('maps weight to thickness in 0.5–5px', () => {
    const now = Date.now();
    const thin = edgeReducer('e1', { weight: 0.0, tier: 'L1Working', ltp_status: 'None' }, { now });
    const thick = edgeReducer('e2', { weight: 1.0, tier: 'L1Working', ltp_status: 'None' }, { now });
    expect(thin.size).toBeCloseTo(0.5, 1);
    expect(thick.size).toBeCloseTo(5.0, 1);
  });

  it('dashes pending-LTP edges, solid for consolidated', () => {
    const now = Date.now();
    const pending = edgeReducer('e1', { weight: 0.5, tier: 'L1Working', ltp_status: 'Pending' }, { now });
    const solid = edgeReducer('e2', { weight: 0.5, tier: 'L1Working', ltp_status: 'Consolidated' }, { now });
    expect(pending.type).toBe('dashed');
    expect(solid.type).toBe('line');
  });

  it('hue differs per tier (L1 warm, L3 cool)', () => {
    const now = Date.now();
    const l1 = edgeReducer('e1', { weight: 0.5, tier: 'L1Working', ltp_status: 'None' }, { now });
    const l3 = edgeReducer('e2', { weight: 0.5, tier: 'L3Semantic', ltp_status: 'None' }, { now });
    expect(parseInt(l1.color.slice(1, 3), 16)).toBeGreaterThan(
      parseInt(l3.color.slice(1, 3), 16)
    );
  });

  it('marks recently-activated edges for pulse animation', () => {
    const now = Date.now();
    const recent = edgeReducer('e1',
      { weight: 0.5, tier: 'L1Working', ltp_status: 'None',
        last_activated: new Date(now - 1000).toISOString() },
      { now });
    const old = edgeReducer('e2',
      { weight: 0.5, tier: 'L1Working', ltp_status: 'None',
        last_activated: new Date(now - 60000).toISOString() },
      { now });
    expect(recent.pulse).toBe(true);
    expect(old.pulse).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/edge-style.test.js 2>&1 | tail -20 && cd -
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`src/handlers/viewer/js/domain/edge-style.js`:

```javascript
const TIER_HUE = {
  L1Working:  '#d33b2c',
  L2Episodic: '#e69f12',
  L3Semantic: '#3c78b5',
};
const DEFAULT_HUE = '#888';
const PULSE_WINDOW_MS = 5000;

/**
 * Map a graph edge's attributes to a sigma reducer result.
 * @param {string} id
 * @param {object} attrs  graphology edge attributes
 * @param {{now:number}} ctx
 * @returns {object} sigma edge spec
 */
export function edgeReducer(id, attrs, ctx) {
  const weight = typeof attrs.weight === 'number' ? attrs.weight : 0.5;
  const size = 0.5 + 4.5 * Math.min(1, Math.max(0, weight));
  const color = TIER_HUE[attrs.tier] || DEFAULT_HUE;

  let type = 'line';
  if (attrs.ltp_status === 'Pending') type = 'dashed';
  else if (attrs.ltp_status === 'JustPromoted') type = 'line'; // extra-bold handled below

  const emphasized = attrs.ltp_status === 'JustPromoted';

  let pulse = false;
  if (attrs.last_activated) {
    const ts = Date.parse(attrs.last_activated);
    if (!Number.isNaN(ts) && (ctx.now - ts) < PULSE_WINDOW_MS) pulse = true;
  }

  return {
    size: emphasized ? size * 1.4 : size,
    color,
    type,
    pulse,
    label: attrs.label || attrs.relation_type || '',
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/edge-style.test.js 2>&1 | tail -20 && cd -
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/handlers/viewer/js/domain/edge-style.js src/handlers/viewer/tests/unit/edge-style.test.js
git commit -m "feat(viewer): edge styling (weight→thickness, tier→hue, LTP→dash, pulse window)"
```

---

### Task 15: Legend component

Static HTML fragment describing the visual vocabulary; rendered in the sidebar. Pure presentation, no logic.

**Files:**
- Create: `src/handlers/viewer/js/ui/legend.js`
- Modify: `src/handlers/viewer/css/style.css` (add `.legend` styles inherited from fork; tweak)

- [ ] **Step 1: Implement directly** (no TDD; single render function, visually validated)

`src/handlers/viewer/js/ui/legend.js`:

```javascript
export function renderLegend(container) {
  container.innerHTML = `
    <div class="legend">
      <h4>Nodes</h4>
      <ul>
        <li><span class="swatch swatch-circle" style="background:#ff6b2c"></span> Memory (working)</li>
        <li><span class="swatch swatch-circle" style="background:#f5b73b"></span> Memory (session)</li>
        <li><span class="swatch swatch-circle" style="background:#4b8bb5"></span> Memory (longterm)</li>
        <li><span class="swatch swatch-square" style="background:#7d74c9"></span> Entity</li>
        <li><span class="swatch swatch-diamond" style="background:#39a887"></span> Episode</li>
      </ul>
      <h4>Edges</h4>
      <ul>
        <li><span class="line line-l1"></span> L1 / Working</li>
        <li><span class="line line-l2"></span> L2 / Episodic</li>
        <li><span class="line line-l3"></span> L3 / Semantic</li>
        <li><span class="line line-dashed"></span> LTP pending</li>
        <li>Thickness ∝ Hebbian weight</li>
        <li>Pulse ⇒ fired in last 5 s</li>
      </ul>
    </div>
  `;
}
```

Append matching swatch styles to `src/handlers/viewer/css/style.css`:

```css
.legend .swatch { display:inline-block; width:10px; height:10px; margin-right:6px; }
.legend .swatch-square { border-radius:0; }
.legend .swatch-circle { border-radius:50%; }
.legend .swatch-diamond { transform:rotate(45deg); }
.legend .line { display:inline-block; width:24px; height:0; border-bottom:2px solid; margin-right:6px; vertical-align:middle; }
.legend .line-l1 { border-color:#d33b2c; }
.legend .line-l2 { border-color:#e69f12; }
.legend .line-l3 { border-color:#3c78b5; }
.legend .line-dashed { border-style:dashed; }
```

- [ ] **Step 2: Sanity-check**

Grep for the function name in the module and verify it's reachable:
`grep -F 'renderLegend' src/handlers/viewer/js/ui/legend.js`

- [ ] **Step 3: Commit**

```bash
git add src/handlers/viewer/js/ui/legend.js src/handlers/viewer/css/style.css
git commit -m "feat(viewer): legend component for node/edge visual vocabulary"
```

---

## Phase 6 — Core Data Flow

### Task 16: API client + GEXF loader with ETag tracking

**Files:**
- Create: `src/handlers/viewer/js/config/api-client.js`
- Create: `src/handlers/viewer/js/graph/loader.js`
- Test: `src/handlers/viewer/tests/unit/api-client.test.js`
- Test: `src/handlers/viewer/tests/unit/loader.test.js`

- [ ] **Step 1: Write `api-client.test.js`**

```javascript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createApiClient } from '../../js/config/api-client.js';

describe('createApiClient', () => {
  let fetchSpy;
  beforeEach(() => { fetchSpy = vi.spyOn(globalThis, 'fetch'); });

  it('attaches X-API-Key header to fetch calls', async () => {
    fetchSpy.mockResolvedValue(new Response('ok', { status: 200 }));
    const api = createApiClient({ baseUrl: 'http://localhost:3000', apiKey: 'abc' });
    await api.fetchGexf('user1');
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe('http://localhost:3000/api/graph/user1/export?format=gexf');
    expect(init.headers['X-API-Key']).toBe('abc');
  });

  it('forwards If-None-Match when etag is provided', async () => {
    fetchSpy.mockResolvedValue(new Response('', { status: 304 }));
    const api = createApiClient({ baseUrl: '', apiKey: 'k' });
    await api.fetchGexf('u', 'W/"abc123"');
    expect(fetchSpy.mock.calls[0][1].headers['If-None-Match']).toBe('W/"abc123"');
  });

  it('builds SSE URL with api_key query param', () => {
    const api = createApiClient({ baseUrl: 'http://localhost:3000', apiKey: 'sse-k' });
    const url = api.sseUrl('u');
    expect(url).toBe('http://localhost:3000/api/events/sse?user_id=u&api_key=sse-k');
  });
});
```

- [ ] **Step 2: Implement `api-client.js`**

```javascript
export function createApiClient({ baseUrl, apiKey }) {
  baseUrl = baseUrl.replace(/\/$/, '');

  async function fetchGexf(userId, etag = null) {
    const headers = { 'X-API-Key': apiKey };
    if (etag) headers['If-None-Match'] = etag;
    const url = `${baseUrl}/api/graph/${encodeURIComponent(userId)}/export?format=gexf`;
    return fetch(url, { headers });
  }

  function sseUrl(userId) {
    const u = `${baseUrl}/api/events/sse?user_id=${encodeURIComponent(userId)}&api_key=${encodeURIComponent(apiKey)}`;
    return u;
  }

  async function fetchMemoryContent(memoryId) {
    const url = `${baseUrl}/api/memories/${encodeURIComponent(memoryId)}`;
    return fetch(url, { headers: { 'X-API-Key': apiKey } });
  }

  return { fetchGexf, sseUrl, fetchMemoryContent };
}
```

- [ ] **Step 3: Write `loader.test.js`**

```javascript
import { describe, it, expect, vi } from 'vitest';
import { createLoader } from '../../js/graph/loader.js';

const SAMPLE_GEXF = `<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" version="1.3">
  <graph defaultedgetype="directed">
    <nodes><node id="n1" label="A"/></nodes>
    <edges/>
  </graph>
</gexf>`;

describe('loader', () => {
  it('parses GEXF text into a graphology Graph', async () => {
    const loader = createLoader({ apiClient: null, gexfParser: (Graph, xml) => {
      const g = new Graph();
      g.addNode('n1', { label: 'A' });
      return g;
    }, GraphClass: class { constructor(){ this.nodes={}; this.addNode = (i,a)=>this.nodes[i]=a; } } });
    const g = await loader.parseFromText(SAMPLE_GEXF);
    expect(Object.keys(g.nodes)).toContain('n1');
  });

  it('fetches GEXF from API and returns {graph, etag, meta}', async () => {
    const apiClient = {
      fetchGexf: vi.fn().mockResolvedValue(new Response(SAMPLE_GEXF, {
        status: 200,
        headers: { etag: 'W/"xyz"', 'content-type': 'application/gexf+xml' },
      })),
    };
    const loader = createLoader({ apiClient,
      gexfParser: () => ({ nodes: { n1: { label: 'A' } } }),
      GraphClass: class {} });
    const result = await loader.fetchFromApi('user1');
    expect(apiClient.fetchGexf).toHaveBeenCalledWith('user1', null);
    expect(result.etag).toBe('W/"xyz"');
    expect(result.graph).toBeDefined();
  });

  it('returns unchanged:true on 304', async () => {
    const apiClient = {
      fetchGexf: vi.fn().mockResolvedValue(new Response(null, { status: 304 })),
    };
    const loader = createLoader({ apiClient, gexfParser: null, GraphClass: null });
    const result = await loader.fetchFromApi('user1', 'W/"abc"');
    expect(result.unchanged).toBe(true);
  });
});
```

- [ ] **Step 4: Implement `loader.js`**

```javascript
export function createLoader({ apiClient, gexfParser, GraphClass }) {
  async function parseFromText(text) {
    return gexfParser(GraphClass, text);
  }

  async function fetchFromApi(userId, prevEtag = null) {
    const resp = await apiClient.fetchGexf(userId, prevEtag);
    if (resp.status === 304) return { unchanged: true };
    if (!resp.ok) throw new Error(`fetch failed: ${resp.status}`);
    const etag = resp.headers.get('etag');
    const text = await resp.text();
    const graph = await parseFromText(text);
    const meta = extractMeta(text);
    return { graph, etag, meta };
  }

  function extractMeta(xml) {
    const m = xml.match(/<server_time>([^<]+)<\/server_time>/);
    return { server_time: m ? m[1] : null };
  }

  return { parseFromText, fetchFromApi };
}
```

- [ ] **Step 5: Run both tests**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/api-client.test.js tests/unit/loader.test.js 2>&1 | tail -20 && cd -
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/handlers/viewer/js/config/api-client.js src/handlers/viewer/js/graph/loader.js src/handlers/viewer/tests/unit/
git commit -m "feat(viewer): API client + GEXF loader with ETag tracking and meta extraction"
```

---

### Task 17: Renderer — sigma bootstrap, reducers, FA2 loading overlay

The fork already has a renderer. Replace its reducer wiring with ours (from Tasks 13–14) and add the FA2 loading overlay per spec.

**Files:**
- Modify: `src/handlers/viewer/js/graph/renderer.js` (inherited from fork; rewrite reducers)

**Layering requirement.** The pure `nodeReducer` / `edgeReducer` from Tasks 13–14 do NOT replace the fork's stateful `nodeReducer` closure (currently in `app.js` around line 776) — they are the *base layer*. The fork's reducer closes over hover/selection/manual-hide state (`hoveredNode`, `selectedNode`, `manuallyHidden`, theme). Wire our reducers as follows:
1. Call `nodeReducer(id, attrs, ctx)` from `node-style.js` to get the base visual spec (`{size, color, shape, haloPulse, recencyBadge, label}`).
2. On top of that, apply the existing state overlays: hover dimming, selected-highlight, manually-hidden recoloring, `graph.areNeighbors()` emphasis. Do NOT drop these behaviors. Carry them from the fork's reducer into the new wrapper.
3. Same pattern for `edgeReducer`.

The result is: pure styling comes from the domain modules (single source of truth, unit-tested); interactive state overlays live in the renderer where they belong.

- [ ] **Step 1: Strip fork's domain reducers, wire in shodh reducers**

Open `src/handlers/viewer/js/graph/renderer.js`. Find the `nodeReducer` and `edgeReducer` registrations (typically inside `new Sigma(graph, container, { nodeReducer: ..., edgeReducer: ... })`). Replace them with a layered wrapper:

```javascript
import { nodeReducer as styleNode } from '../domain/node-style.js';
import { edgeReducer as styleEdge } from '../domain/edge-style.js';

export function mount(graph, container, opts = {}) {
  // State owned by the renderer, mutated by interaction handlers.
  const state = { hoveredNode: null, selectedNode: null, manuallyHidden: new Set() };

  const sigma = new Sigma(graph, container, {
    nodeReducer: (id, attrs) => {
      const base = styleNode(id, attrs, { now: Date.now() });
      // Layer interactive state on top of pure styling.
      if (state.manuallyHidden.has(id)) return { ...base, hidden: true };
      if (state.hoveredNode && state.hoveredNode !== id && !graph.areNeighbors(state.hoveredNode, id)) {
        return { ...base, color: 'rgba(0,0,0,0.1)', label: '', zIndex: 0 };
      }
      if (state.hoveredNode === id || state.selectedNode === id) {
        return { ...base, highlighted: true, zIndex: 1 };
      }
      return base;
    },
    edgeReducer: (id, attrs) => {
      const base = styleEdge(id, attrs, { now: Date.now() });
      if (state.hoveredNode) {
        const [s, t] = graph.extremities(id);
        const touches = s === state.hoveredNode || t === state.hoveredNode;
        if (!touches) return { ...base, hidden: true };
      }
      return base;
    },
    ...opts,
  });

  // FA2 loading overlay
  const overlay = document.createElement('div');
  overlay.className = 'fa2-overlay';
  overlay.textContent = 'Laying out…';
  container.appendChild(overlay);

  // Hide overlay after FA2 converges (best-effort: hide on first stall, or 5s timeout)
  const fa2Worker = startFa2(graph);
  let lastTick = performance.now();
  const checkConverged = setInterval(() => {
    if (performance.now() - lastTick > 1500) {
      overlay.remove();
      clearInterval(checkConverged);
      fa2Worker.stop();
    }
  }, 500);
  setTimeout(() => {
    overlay.remove();
    clearInterval(checkConverged);
  }, 10_000);

  // Animation loop: pulses require refresh
  let rafId;
  function tick() {
    sigma.refresh();
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);

  return { sigma, stop: () => { cancelAnimationFrame(rafId); fa2Worker.stop(); } };
}

function startFa2(graph) {
  // FA2 worker from graphology-layout-forceatlas2 — specifics depend on fork's choice
  // See src/handlers/viewer/js/graph/layout.js for the worker wrapper
  return { stop: () => {} }; // placeholder; wire real FA2 worker here
}
```

Add CSS in `style.css`:

```css
.fa2-overlay {
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  padding: 8px 16px; background: rgba(0,0,0,0.6);
  color: white; border-radius: 4px; font-size: 14px;
  pointer-events: none;
}
```

- [ ] **Step 2: Verify the existing FA2 worker integration point**

Open `src/handlers/viewer/js/graph/layout.js` (inherited from fork). Confirm it exports a function starting a web worker that mutates node positions. If the interface differs, adapt the `startFa2` placeholder above.

If missing, copy the worker pattern from `graphology-layout-forceatlas2` docs and land it as part of this task. Keep the implementation minimal: start worker, receive position updates, expose `.stop()`.

- [ ] **Step 3: Smoke-check with a hand-built graph**

Write `src/handlers/viewer/tests/unit/renderer.smoke.js`:

```javascript
// Smoke test only — full DOM/WebGL rendering verified in Playwright.
import { describe, it, expect, vi } from 'vitest';

describe('renderer wiring', () => {
  it('exports a mount function', async () => {
    const mod = await import('../../js/graph/renderer.js');
    expect(typeof mod.mount).toBe('function');
  });
});
```

- [ ] **Step 4: Run smoke test**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/renderer.smoke.js 2>&1 | tail -10 && cd -
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/handlers/viewer/js/graph/renderer.js src/handlers/viewer/js/graph/layout.js src/handlers/viewer/css/style.css src/handlers/viewer/tests/unit/renderer.smoke.js
git commit -m "feat(viewer): shodh reducers + FA2 loading overlay in renderer"
```

---

## Phase 7 — Live Mode

### Task 18: SSE client with exponential backoff reconnect

**Files:**
- Create: `src/handlers/viewer/js/live/sse-client.js`
- Test: `src/handlers/viewer/tests/unit/sse-client.test.js`

- [ ] **Step 1: Write the failing test**

```javascript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createSseClient } from '../../js/live/sse-client.js';

class MockEventSource {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.listeners = {};
    MockEventSource.instances.push(this);
    setTimeout(() => { this.readyState = 1; this._fire('open', {}); }, 0);
  }
  addEventListener(evt, fn) { (this.listeners[evt] ||= []).push(fn); }
  close() { this.readyState = 2; }
  _fire(evt, data) { (this.listeners[evt] || []).forEach(f => f(data)); }
}

describe('sseClient', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockEventSource.instances = [];
    globalThis.EventSource = MockEventSource;
  });
  afterEach(() => vi.useRealTimers());

  it('connects to the given URL and calls onMessage for each event', async () => {
    const onMessage = vi.fn();
    const client = createSseClient({
      url: 'http://x/sse?user_id=u&api_key=k',
      onMessage,
    });
    client.connect();
    await vi.runOnlyPendingTimersAsync();
    const es = MockEventSource.instances[0];
    es._fire('message', { data: '{"event_type":"TODO_CREATED"}' });
    expect(onMessage).toHaveBeenCalledTimes(1);
  });

  it('reconnects with exponential backoff on error', async () => {
    const client = createSseClient({ url: 'http://x/sse', onMessage: vi.fn() });
    client.connect();
    await vi.runOnlyPendingTimersAsync();
    MockEventSource.instances[0]._fire('error', {});
    expect(MockEventSource.instances.length).toBe(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(MockEventSource.instances.length).toBe(2);
    MockEventSource.instances[1]._fire('error', {});
    await vi.advanceTimersByTimeAsync(2000);
    expect(MockEventSource.instances.length).toBe(3);
  });

  it('caps backoff at 30s', async () => {
    const client = createSseClient({ url: 'http://x/sse', onMessage: vi.fn() });
    client.connect();
    await vi.runOnlyPendingTimersAsync();
    // Trip six errors to exceed 30s cap
    for (let i = 0; i < 6; i++) {
      const cur = MockEventSource.instances[MockEventSource.instances.length - 1];
      cur._fire('error', {});
      await vi.advanceTimersByTimeAsync(30_000);
    }
    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(6);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/sse-client.test.js 2>&1 | tail -10 && cd -
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```javascript
export function createSseClient({ url, onMessage, onStatusChange = () => {} }) {
  let es = null;
  let attempts = 0;
  let reconnectTimer = null;
  let stopped = false;

  function connect() {
    if (stopped) return;
    onStatusChange('connecting');
    es = new EventSource(url);
    es.addEventListener('open', () => {
      attempts = 0;
      onStatusChange('connected');
    });
    es.addEventListener('message', (evt) => {
      try {
        onMessage(JSON.parse(evt.data));
      } catch (e) {
        onMessage({ raw: evt.data });
      }
    });
    es.addEventListener('error', () => {
      onStatusChange('disconnected');
      es.close();
      scheduleReconnect();
    });
  }

  function scheduleReconnect() {
    if (stopped) return;
    const delay = Math.min(30_000, 1000 * Math.pow(2, attempts));
    attempts++;
    reconnectTimer = setTimeout(connect, delay);
  }

  function close() {
    stopped = true;
    if (es) es.close();
    if (reconnectTimer) clearTimeout(reconnectTimer);
    onStatusChange('closed');
  }

  return { connect, close };
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/sse-client.test.js 2>&1 | tail -10 && cd -
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/handlers/viewer/js/live/sse-client.js src/handlers/viewer/tests/unit/sse-client.test.js
git commit -m "feat(viewer): SSE client with exponential backoff reconnect (1-2-4-8-16-30s)"
```

---

### Task 19: Refetch — debounce, ETag short-circuit, graph-diff, in-place mutate

**Files:**
- Create: `src/handlers/viewer/js/live/refetch.js`
- Test: `src/handlers/viewer/tests/unit/refetch.test.js`

- [ ] **Step 1: Write failing tests**

```javascript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createRefetcher, diffGraphs } from '../../js/live/refetch.js';

describe('diffGraphs', () => {
  it('computes added / removed / common nodes', () => {
    const prev = { nodes: { a: {}, b: {} }, edges: {} };
    const next = { nodes: { b: {}, c: {} }, edges: {} };
    const d = diffGraphs(prev, next);
    expect(d.addedNodes).toEqual(['c']);
    expect(d.removedNodes).toEqual(['a']);
    expect(d.commonNodes).toEqual(['b']);
  });
  it('same for edges', () => {
    const prev = { nodes: {}, edges: { x: {}, y: {} } };
    const next = { nodes: {}, edges: { y: {}, z: {} } };
    const d = diffGraphs(prev, next);
    expect(d.addedEdges).toEqual(['z']);
    expect(d.removedEdges).toEqual(['x']);
    expect(d.commonEdges).toEqual(['y']);
  });
});

describe('createRefetcher', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => vi.useRealTimers());

  it('debounces trigger calls by 2000ms', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({ unchanged: true });
    const r = createRefetcher({ fetchImpl, applyDiff: vi.fn(), debounceMs: 2000 });
    r.trigger(); r.trigger(); r.trigger();
    expect(fetchImpl).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1999);
    expect(fetchImpl).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(2);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('skips applyDiff when response is unchanged (304)', async () => {
    const applyDiff = vi.fn();
    const fetchImpl = vi.fn().mockResolvedValue({ unchanged: true });
    const r = createRefetcher({ fetchImpl, applyDiff, debounceMs: 0 });
    r.trigger();
    await vi.runAllTimersAsync();
    expect(applyDiff).not.toHaveBeenCalled();
  });

  it('applies diff on fresh response', async () => {
    const applyDiff = vi.fn();
    const fresh = { graph: { nodes: { a: {} }, edges: {} }, etag: 'W/"1"' };
    const fetchImpl = vi.fn().mockResolvedValue(fresh);
    const r = createRefetcher({ fetchImpl, applyDiff, debounceMs: 0 });
    r.trigger();
    await vi.runAllTimersAsync();
    expect(applyDiff).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/refetch.test.js 2>&1 | tail -10 && cd -
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```javascript
export function diffGraphs(prev, next) {
  const prevNodeIds = new Set(Object.keys(prev.nodes || {}));
  const nextNodeIds = new Set(Object.keys(next.nodes || {}));
  const prevEdgeIds = new Set(Object.keys(prev.edges || {}));
  const nextEdgeIds = new Set(Object.keys(next.edges || {}));

  const addedNodes = [...nextNodeIds].filter(i => !prevNodeIds.has(i));
  const removedNodes = [...prevNodeIds].filter(i => !nextNodeIds.has(i));
  const commonNodes = [...nextNodeIds].filter(i => prevNodeIds.has(i));
  const addedEdges = [...nextEdgeIds].filter(i => !prevEdgeIds.has(i));
  const removedEdges = [...prevEdgeIds].filter(i => !nextEdgeIds.has(i));
  const commonEdges = [...nextEdgeIds].filter(i => prevEdgeIds.has(i));

  return { addedNodes, removedNodes, commonNodes, addedEdges, removedEdges, commonEdges };
}

export function applyDiffToGraph(sigma, graph, next, diff) {
  // In-place mutate the graphology Graph
  for (const id of diff.removedNodes) graph.dropNode(id);
  for (const id of diff.addedNodes) {
    const attrs = next.nodes[id];
    // Spawn near a neighbor if possible, else origin + jitter
    const neighbor = findAnyNeighborId(next, id);
    const base = neighbor && graph.hasNode(neighbor)
      ? graph.getNodeAttributes(neighbor)
      : { x: 0, y: 0 };
    const jitter = () => (Math.random() - 0.5) * 20;
    graph.addNode(id, { ...attrs, x: (base.x || 0) + jitter(), y: (base.y || 0) + jitter() });
  }
  for (const id of diff.commonNodes) graph.mergeNodeAttributes(id, next.nodes[id]);
  for (const id of diff.removedEdges) graph.dropEdge(id);
  for (const id of diff.addedEdges) {
    const e = next.edges[id];
    graph.addEdgeWithKey(id, e.source, e.target, e);
  }
  for (const id of diff.commonEdges) graph.mergeEdgeAttributes(id, next.edges[id]);
  sigma.refresh();
}

function findAnyNeighborId(next, nodeId) {
  for (const [_, e] of Object.entries(next.edges || {})) {
    if (e.source === nodeId) return e.target;
    if (e.target === nodeId) return e.source;
  }
  return null;
}

export function createRefetcher({ fetchImpl, applyDiff, debounceMs = 2000 }) {
  let timer = null;
  let inFlight = false;
  let pending = false;

  async function run() {
    if (inFlight) { pending = true; return; }
    inFlight = true;
    try {
      const result = await fetchImpl();
      if (!result.unchanged) applyDiff(result);
    } catch (e) {
      console.error('[refetch] failed', e);
    } finally {
      inFlight = false;
      if (pending) { pending = false; schedule(); }
    }
  }

  function schedule() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(run, debounceMs);
  }

  function trigger() {
    if (inFlight) { pending = true; return; }
    schedule();
  }

  return { trigger };
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/refetch.test.js 2>&1 | tail -10 && cd -
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/handlers/viewer/js/live/refetch.js src/handlers/viewer/tests/unit/refetch.test.js
git commit -m "feat(viewer): debounced refetch with graph-diff and in-place mutate"
```

---

## Phase 8 — UI

### Task 20: Filter sidebar — tier, type, weight, activation, LTP, recency

Fork provides a sidebar scaffold. Replace its filter set with shodh's filters.

**Files:**
- Modify: `src/handlers/viewer/js/ui/sidebar.js` (inherited — replace filter controls)
- Create: `src/handlers/viewer/js/domain/filters.js` (pure logic)
- Test: `src/handlers/viewer/tests/unit/filters.test.js`

- [ ] **Step 1: Write the failing test for pure filter logic**

```javascript
import { describe, it, expect } from 'vitest';
import { matchesFilters } from '../../js/domain/filters.js';

const node = (overrides = {}) => ({
  type: 'memory', tier: 'Working', activation: 0.3,
  last_accessed: new Date().toISOString(), ...overrides,
});
const edge = (overrides = {}) => ({
  tier: 'L1Working', weight: 0.4, ltp_status: 'None',
  last_activated: new Date().toISOString(), ...overrides,
});

describe('matchesFilters (node)', () => {
  it('includes node when its tier is in activeTiers', () => {
    const f = { activeTiers: new Set(['Working']), activeTypes: new Set(['memory']),
                minActivation: 0 };
    expect(matchesFilters.node(node(), f)).toBe(true);
  });
  it('excludes node when tier is hidden', () => {
    const f = { activeTiers: new Set(['Longterm']), activeTypes: new Set(['memory']),
                minActivation: 0 };
    expect(matchesFilters.node(node({ tier: 'Working' }), f)).toBe(false);
  });
  it('excludes node below minActivation', () => {
    const f = { activeTiers: new Set(['Working']), activeTypes: new Set(['memory']),
                minActivation: 0.5 };
    expect(matchesFilters.node(node({ activation: 0.2 }), f)).toBe(false);
  });
});

describe('matchesFilters (edge)', () => {
  it('excludes edge below minWeight', () => {
    const f = { activeTiers: new Set(['L1Working']), minWeight: 0.5, activeLtp: new Set(['None']) };
    expect(matchesFilters.edge(edge({ weight: 0.2 }), f)).toBe(false);
  });
  it('includes edge matching tier + LTP + weight', () => {
    const f = { activeTiers: new Set(['L1Working']), minWeight: 0.3, activeLtp: new Set(['None']) };
    expect(matchesFilters.edge(edge(), f)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/filters.test.js 2>&1 | tail -10 && cd -
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`src/handlers/viewer/js/domain/filters.js`:

```javascript
export const matchesFilters = {
  node(attrs, f) {
    if (!f.activeTypes.has(attrs.type)) return false;
    if (attrs.type === 'memory') {
      if (!f.activeTiers.has(attrs.tier)) return false;
    }
    if ((attrs.activation || 0) < (f.minActivation || 0)) return false;
    if (f.recencyWindowMs != null && attrs.last_accessed) {
      const age = Date.now() - Date.parse(attrs.last_accessed);
      if (age > f.recencyWindowMs) return false;
    }
    return true;
  },

  edge(attrs, f) {
    if (!f.activeTiers.has(attrs.tier)) return false;
    if ((attrs.weight || 0) < (f.minWeight || 0)) return false;
    if (!f.activeLtp.has(attrs.ltp_status || 'None')) return false;
    return true;
  },
};
```

- [ ] **Step 4: Wire filters into renderer reducers**

Modify `src/handlers/viewer/js/graph/renderer.js` to accept a filter-state accessor in `mount(graph, container, opts)`:

```javascript
const filterState = opts.filterState || (() => ({
  activeTiers: new Set(['Working', 'Session', 'Longterm', 'L1Working', 'L2Episodic', 'L3Semantic']),
  activeTypes: new Set(['memory', 'entity', 'episode']),
  activeLtp: new Set(['None', 'Pending', 'Consolidated', 'JustPromoted']),
  minActivation: 0, minWeight: 0, recencyWindowMs: null,
}));

const sigma = new Sigma(graph, container, {
  nodeReducer: (id, attrs) => {
    const f = filterState();
    if (!matchesFilters.node(attrs, f)) return { hidden: true };
    return nodeReducer(id, attrs, { now: Date.now() });
  },
  edgeReducer: (id, attrs) => {
    const f = filterState();
    if (!matchesFilters.edge(attrs, f)) return { hidden: true };
    return edgeReducer(id, attrs, { now: Date.now() });
  },
});
```

- [ ] **Step 5: Replace sidebar controls**

In `src/handlers/viewer/js/ui/sidebar.js`, replace the fork's filter UI with shodh filter controls. Render one checkbox per tier (3 memory tiers + 3 edge tiers), one per type, one slider for min-weight, one for min-activation, one dropdown for LTP status, and one select for recency window. On change, call `onFilterChange(newState)` supplied by the caller.

Concretely the module exports:

```javascript
export function renderSidebar(container, { onFilterChange, stats }) {
  // ... render filter controls + stats panel + live indicator ...
  // onFilterChange({activeTiers, activeTypes, activeLtp, minActivation, minWeight, recencyWindowMs})
}
```

- [ ] **Step 6: Run tests**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/filters.test.js 2>&1 | tail -10 && cd -
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/handlers/viewer/js/domain/filters.js src/handlers/viewer/js/ui/sidebar.js src/handlers/viewer/js/graph/renderer.js src/handlers/viewer/tests/unit/filters.test.js
git commit -m "feat(viewer): filter set — tier/type/weight/activation/LTP/recency"
```

---

### Task 21: Detail panel with lazy content fetch

**Files:**
- Modify: `src/handlers/viewer/js/ui/detail-panel.js` (inherited — adapt to shodh attrs + lazy fetch)

- [ ] **Step 1: Implement directly** (presentation; covered by Playwright in Task 25)

```javascript
export function createDetailPanel({ container, apiClient }) {
  async function show(graph, id) {
    const attrs = graph.getNodeAttributes(id);
    container.innerHTML = `
      <h3>${escape(attrs.label || id)}</h3>
      <div class="meta">${escape(attrs.type)} · ${escape(attrs.tier || '')}</div>
      <dl>
        ${kv('importance', attrs.importance)}
        ${kv('activation', attrs.activation)}
        ${kv('access_count', attrs.access_count)}
        ${kv('last_accessed', attrs.last_accessed)}
      </dl>
      <div class="content">${escape(attrs.content || '')}</div>
    `;

    if (!attrs.content && attrs.type === 'memory') {
      try {
        const resp = await apiClient.fetchMemoryContent(id);
        if (resp.ok) {
          const body = await resp.json();
          const full = body.content || body.experience?.content;
          if (full) {
            container.querySelector('.content').textContent = full;
            graph.mergeNodeAttributes(id, { content: full });
          }
        } else {
          container.querySelector('.content').textContent = 'Full content unavailable.';
        }
      } catch (e) {
        container.querySelector('.content').textContent = 'Full content unavailable.';
      }
    }
  }

  function hide() { container.innerHTML = ''; }
  return { show, hide };
}

function kv(label, value) {
  if (value == null) return '';
  return `<dt>${escape(label)}</dt><dd>${escape(String(value))}</dd>`;
}
function escape(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}
```

- [ ] **Step 2: Commit**

```bash
git add src/handlers/viewer/js/ui/detail-panel.js
git commit -m "feat(viewer): detail panel with lazy /api/memories/{id} content fetch"
```

---

### Task 22: Curation highlights — weakness, orphans, dead edges

**Files:**
- Create: `src/handlers/viewer/js/domain/curation.js`
- Modify: `src/handlers/viewer/js/ui/sidebar.js` (add curation toggles)
- Test: `src/handlers/viewer/tests/unit/curation.test.js`

- [ ] **Step 1: Write failing test**

```javascript
import { describe, it, expect } from 'vitest';
import { isWeak, isOrphan, isDeadEdge } from '../../js/domain/curation.js';

describe('curation', () => {
  it('weak = importance < 0.2 AND access_count < 5', () => {
    expect(isWeak({ importance: 0.1, access_count: 2 })).toBe(true);
    expect(isWeak({ importance: 0.1, access_count: 6 })).toBe(false);
    expect(isWeak({ importance: 0.3, access_count: 2 })).toBe(false);
  });
  it('orphan = zero-degree', () => {
    const graph = { degree: (id) => id === 'a' ? 0 : 3 };
    expect(isOrphan(graph, 'a')).toBe(true);
    expect(isOrphan(graph, 'b')).toBe(false);
  });
  it('deadEdge = last_activated > 7d OR activation_count == 0', () => {
    const now = Date.now();
    const tenDaysAgo = new Date(now - 10*24*60*60*1000).toISOString();
    const oneDayAgo = new Date(now - 24*60*60*1000).toISOString();
    expect(isDeadEdge({ last_activated: tenDaysAgo, activation_count: 2 }, now)).toBe(true);
    expect(isDeadEdge({ last_activated: oneDayAgo, activation_count: 0 }, now)).toBe(true);
    expect(isDeadEdge({ last_activated: oneDayAgo, activation_count: 2 }, now)).toBe(false);
  });
});
```

- [ ] **Step 2: Implement**

`src/handlers/viewer/js/domain/curation.js`:

```javascript
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

export function isWeak(attrs) {
  return (attrs.importance || 0) < 0.2 && (attrs.access_count || 0) < 5;
}

export function isOrphan(graph, id) {
  return graph.degree(id) === 0;
}

export function isDeadEdge(attrs, now = Date.now()) {
  if ((attrs.activation_count || 0) === 0) return true;
  if (attrs.last_activated) {
    const ts = Date.parse(attrs.last_activated);
    if (!Number.isNaN(ts) && (now - ts) > WEEK_MS) return true;
  }
  return false;
}
```

Wire into the renderer by adding a "curation-mode" toggle to sidebar. When active, the node/edge reducer paints the predicates with a red ring / dotted overlay. Set `attrs.__curation` in a separate reducer pass rather than mutating the graph.

- [ ] **Step 3: Run test**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/curation.test.js 2>&1 | tail -10 && cd -
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/handlers/viewer/js/domain/curation.js src/handlers/viewer/js/ui/sidebar.js src/handlers/viewer/tests/unit/curation.test.js
git commit -m "feat(viewer): curation highlights — weakness, orphans, dead edges"
```

---

### Task 23: Export filtered subgraph as GEXF + export IDs as text

**Files:**
- Create: `src/handlers/viewer/js/domain/export.js`
- Modify: `src/handlers/viewer/js/ui/sidebar.js` (add export buttons)
- Test: `src/handlers/viewer/tests/unit/export.test.js`

- [ ] **Step 1: Write failing test**

```javascript
import { describe, it, expect, vi } from 'vitest';
import { exportIdsAsText, exportVisibleAsGexf } from '../../js/domain/export.js';

describe('exportIdsAsText', () => {
  it('joins IDs with newlines', () => {
    const result = exportIdsAsText(['a', 'b', 'c']);
    expect(result).toBe('a\nb\nc\n');
  });
});

describe('exportVisibleAsGexf', () => {
  it('delegates to gexfWrite with a filtered subgraph', () => {
    const gexfWrite = vi.fn().mockReturnValue('<gexf/>');
    const SubgraphClass = class {
      constructor() { this.nodes = []; this.edges = []; }
      addNode(id, a) { this.nodes.push([id, a]); }
      addEdge(s, t, a) { this.edges.push([s, t, a]); }
    };
    const graph = {
      nodes: () => ['a', 'b', 'hidden'],
      edges: () => ['ab'],
      getNodeAttributes: (i) => ({ id: i }),
      getEdgeAttributes: () => ({}),
      source: () => 'a', target: () => 'b',
    };
    const isVisible = (id) => id !== 'hidden';
    const result = exportVisibleAsGexf(graph, SubgraphClass, gexfWrite, isVisible);
    expect(result).toBe('<gexf/>');
    expect(gexfWrite).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Implement**

```javascript
export function exportIdsAsText(ids) {
  return ids.map(i => String(i)).join('\n') + '\n';
}

export function exportVisibleAsGexf(graph, SubgraphClass, gexfWrite, isNodeVisible) {
  const sub = new SubgraphClass();
  for (const id of graph.nodes()) {
    if (isNodeVisible(id)) sub.addNode(id, graph.getNodeAttributes(id));
  }
  for (const id of graph.edges()) {
    const s = graph.source(id), t = graph.target(id);
    if (isNodeVisible(s) && isNodeVisible(t)) {
      sub.addEdge(s, t, graph.getEdgeAttributes(id));
    }
  }
  return gexfWrite(sub);
}

export function download(text, filename, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
```

Wire into sidebar:

```javascript
// In sidebar.js, add two buttons:
exportSubgraphBtn.onclick = () => {
  const xml = exportVisibleAsGexf(graph, Graph, graphologyLibrary.gexf.write, isNodeVisible);
  download(xml, 'subgraph.gexf', 'application/gexf+xml');
};
exportIdsBtn.onclick = () => {
  const ids = selection.currentIds();
  download(exportIdsAsText(ids), 'ids.txt', 'text/plain');
};
```

- [ ] **Step 3: Run test**

```bash
cd src/handlers/viewer && npx vitest run tests/unit/export.test.js 2>&1 | tail -10 && cd -
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/handlers/viewer/js/domain/export.js src/handlers/viewer/js/ui/sidebar.js src/handlers/viewer/tests/unit/export.test.js
git commit -m "feat(viewer): export filtered subgraph (GEXF) + selected IDs (text)"
```

---

## Phase 9 — Boot & Validation

### Task 24: Boot module — URL parsing, mode detection, wire-up

**Files:**
- Modify: `src/handlers/viewer/js/boot.js` (rewrite from fork's version — shodh-specific)
- Modify: `src/handlers/viewer/index.html` (entry script tag)

- [ ] **Step 1: Implement**

`src/handlers/viewer/js/boot.js`:

```javascript
import { createApiClient } from './config/api-client.js';
import { createLoader } from './graph/loader.js';
import { mount } from './graph/renderer.js';
import { createSseClient } from './live/sse-client.js';
import { createRefetcher, diffGraphs, applyDiffToGraph } from './live/refetch.js';
import { renderSidebar } from './ui/sidebar.js';
import { renderLegend } from './ui/legend.js';
import { createDetailPanel } from './ui/detail-panel.js';

function detectMode() {
  const params = new URLSearchParams(window.location.search);
  const userId = params.get('user_id') || window.SHODH_USER_ID;
  const file = params.get('file');
  if (file) return { mode: 'snapshot-remote', file };
  if (userId && userId !== 'default') return { mode: 'live', userId };
  return { mode: 'drop-zone' };
}

async function main() {
  const mode = detectMode();
  const apiClient = createApiClient({
    baseUrl: window.location.origin,
    apiKey: window.SHODH_API_KEY,
  });
  const loader = createLoader({
    apiClient,
    gexfParser: (G, xml) => graphologyLibrary.gexf.parse(G, xml),
    GraphClass: graphology.Graph,
  });

  const container = document.getElementById('graph-container');
  const sidebarEl = document.getElementById('sidebar');
  const legendEl = document.getElementById('legend');
  const detailEl = document.getElementById('detail');

  renderLegend(legendEl);

  if (mode.mode === 'live') {
    let prevEtag = null;
    const first = await loader.fetchFromApi(mode.userId);
    if (first.unchanged) throw new Error('initial fetch returned 304?');
    prevEtag = first.etag;
    const { sigma } = mount(first.graph, container);
    const detail = createDetailPanel({ container: detailEl, apiClient });
    sigma.on('clickNode', ({ node }) => detail.show(first.graph, node));

    const refetcher = createRefetcher({
      fetchImpl: async () => {
        const r = await loader.fetchFromApi(mode.userId, prevEtag);
        if (!r.unchanged) prevEtag = r.etag;
        return r;
      },
      applyDiff: (r) => {
        const next = { nodes: nodesOf(r.graph), edges: edgesOf(r.graph) };
        const prev = { nodes: nodesOf(first.graph), edges: edgesOf(first.graph) };
        applyDiffToGraph(sigma, first.graph, next, diffGraphs(prev, next));
      },
      debounceMs: 2000,
    });

    const sse = createSseClient({
      url: apiClient.sseUrl(mode.userId),
      onMessage: () => refetcher.trigger(),
      onStatusChange: (s) => document.getElementById('sse-status').textContent = s,
    });
    sse.connect();

    renderSidebar(sidebarEl, {
      onFilterChange: (f) => { sigma.refresh(); /* reducers read filter state on each refresh */ },
      stats: { nodes: first.graph.order, edges: first.graph.size },
    });
  } else if (mode.mode === 'snapshot-remote') {
    const resp = await fetch(mode.file);
    const text = await resp.text();
    const graph = await loader.parseFromText(text);
    mount(graph, container);
  } else {
    renderDropZone(container, async (text) => {
      const graph = await loader.parseFromText(text);
      mount(graph, container);
    });
  }
}

function nodesOf(g) {
  const out = {};
  g.forEachNode((id, a) => { out[id] = a; });
  return out;
}
function edgesOf(g) {
  const out = {};
  g.forEachEdge((id, a, src, tgt) => { out[id] = { ...a, source: src, target: tgt }; });
  return out;
}
function renderDropZone(container, onText) {
  container.innerHTML = `<div class="drop-zone">Drop a .gexf file here</div>`;
  container.addEventListener('dragover', (e) => e.preventDefault());
  container.addEventListener('drop', async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const text = await file.text();
    onText(text);
  });
}

main().catch(e => {
  console.error('[boot] failed', e);
  document.body.innerHTML = `<div class="banner">Failed to load: ${e.message}</div>`;
});
```

Update `src/handlers/viewer/index.html` to import `boot.js` as module:

```html
<script type="module" nonce="{{NONCE}}" src="/graph/viewer/js/boot.js"></script>
```

Importmap at top of `<head>`:

```html
<script type="importmap" nonce="{{NONCE}}">
{
  "imports": {
    "graphology": "/graph/assets/graphology.umd.min.js",
    "graphology-gexf": "/graph/assets/graphology-gexf.umd.min.js",
    "graphology-layout-forceatlas2": "/graph/assets/graphology-layout-forceatlas2.umd.min.js",
    "sigma": "/graph/assets/sigma.min.js"
  }
}
</script>
```

(UMD bundles don't need importmap entries to work via globals, but include the map so the source code can use `import` syntax consistently. Adjust imports in modules if UMD requires globals-only access.)

- [ ] **Step 2: Compile-check**

Run: `cargo check --lib 2>&1 | tail -10`
Expected: no errors (only JS/HTML changed; Rust unaffected except compile-time embed may notice added files).

- [ ] **Step 3: Commit**

```bash
git add src/handlers/viewer/js/boot.js src/handlers/viewer/index.html
git commit -m "feat(viewer): boot module with URL-mode detection and wire-up"
```

---

### Task 25: Playwright smoke test + manual smoke checklist

**Files:**
- Create: `src/handlers/viewer/tests/e2e/smoke.spec.js`
- Create: `src/handlers/viewer/tests/fixtures/tiny.gexf`
- Modify: `src/handlers/viewer/package.json` (add playwright)

- [ ] **Step 1: Install Playwright**

```bash
cd src/handlers/viewer && npm install --save-dev @playwright/test && npx playwright install chromium && cd -
```

- [ ] **Step 2: Create the fixture**

`src/handlers/viewer/tests/fixtures/tiny.gexf`: a 10-node GEXF with one of each type + one edge per tier + one pending LTP. Hand-craft it by copying a recent real export and trimming. Verify it round-trips through `graphology.gexf.parse` using a throwaway Node script.

- [ ] **Step 3: Write the Playwright spec**

`src/handlers/viewer/tests/e2e/smoke.spec.js`:

```javascript
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.SHODH_BASE_URL || 'http://localhost:3000';
const API_KEY = process.env.SHODH_DEV_API_KEY || 'test-key';
const USER_ID = process.env.SHODH_TEST_USER || 'smoke-user';

test.describe('viewer smoke', () => {
  test('renders canvas and shows legend', async ({ page }) => {
    await page.goto(`${BASE_URL}/graph/view2?user_id=${USER_ID}`);
    await expect(page.locator('canvas')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('.legend')).toBeVisible();
  });

  test('opens detail panel on node click', async ({ page }) => {
    await page.goto(`${BASE_URL}/graph/view2?user_id=${USER_ID}`);
    await page.waitForSelector('canvas');
    // sigma renders nodes as WebGL — can't click a node by locator.
    // Fall back to synthesizing a click via sigma's API exposed on window for testing.
    await page.evaluate(() => window.__testHooks.clickFirstNode());
    await expect(page.locator('#detail h3')).toBeVisible();
  });

  test('SSE-driven refetch within 5s', async ({ page }) => {
    await page.goto(`${BASE_URL}/graph/view2?user_id=${USER_ID}`);
    await page.waitForSelector('canvas');
    const before = await page.evaluate(() => window.__testHooks.graphOrder());

    // Trigger a write. Requires a test-only API endpoint or direct DB write;
    // use a tiny helper at /api/test/seed-memory?user_id=... that the test
    // harness enables only when SHODH_TEST_MODE=1.
    await fetch(`${BASE_URL}/api/test/seed-memory?user_id=${USER_ID}`, {
      headers: { 'X-API-Key': API_KEY },
      method: 'POST',
    });

    await page.waitForFunction(
      (prev) => window.__testHooks.graphOrder() > prev,
      before,
      { timeout: 5000 }
    );
  });
});
```

Add `window.__testHooks = { clickFirstNode(), graphOrder() }` helpers to `boot.js` behind a `window.location.search.includes('test=1')` or `SHODH_TEST_MODE` gate.

- [ ] **Step 4: Add the manual smoke checklist**

Create `src/handlers/viewer/SMOKE.md` with the 4-step checklist from the spec:

```markdown
# Manual Smoke Test (run before migration flip)

1. Load a real shodh instance with >1k nodes at `/graph/view2?user_id=<you>`. Viewer opens, renders within 5s.
2. Trigger a burst of activity (50 recalls in a minute) via `shodh recall ...`. Pulses appear on active edges; no layout jitter.
3. Close and reopen mid-session. Reconnect works; state consistent.
4. Drag-drop a snapshot `.gexf` into the canvas. SSE detaches. Graph replaced.
```

- [ ] **Step 5: Commit**

```bash
git add src/handlers/viewer/tests/e2e/smoke.spec.js src/handlers/viewer/tests/fixtures/tiny.gexf src/handlers/viewer/SMOKE.md src/handlers/viewer/package.json src/handlers/viewer/package-lock.json
git commit -m "test(viewer): Playwright smoke spec + manual smoke checklist"
```

---

## Final Verification

- [ ] **Run the full test suites**

```bash
cargo test --lib 2>&1 | tail -20
cd src/handlers/viewer && npx vitest run 2>&1 | tail -20 && cd -
```
Expected: all green.

- [ ] **Clippy across the workspace**

Run: `cargo clippy --lib -- -D warnings 2>&1 | tail -20`
Expected: no warnings.

- [ ] **Open a PR**

```bash
git push -u origin feature/gexf-viewer
gh pr create --title "GEXF viewer — shodh-semantic graph browser" --body "$(cat <<'EOF'
## Summary
- Ships `GET /graph/view2` — browser-based sigma.js viewer targeting ~50k nodes
- GEXF export gains edge/node/episode attrs, server_time, ETag, ?include_content
- /api/events/sse accepts ?api_key= query param so browser EventSource can auth
- Live mode: SSE-triggered refetch with 2s debounce, If-None-Match short-circuit, in-place graph-diff+mutate
- Read-only curation: weakness/orphan/dead-edge highlights, subgraph + ID exports

## Test plan
- [ ] `cargo test --lib handlers::export::tests` passes
- [ ] `cargo test --lib auth::tests` passes
- [ ] `cd src/handlers/viewer && npx vitest run` passes
- [ ] `cd src/handlers/viewer && npx playwright test` passes against a running shodh
- [ ] Manual smoke (src/handlers/viewer/SMOKE.md) passes against a real shodh instance with >1k nodes

## Spec + Plan
- Design: `docs/specs/2026-04-16-gexf-viewer-design.md`
- Plan: `docs/specs/2026-04-16-gexf-viewer-plan.md`
EOF
)"
```
