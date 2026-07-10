//! LIVE demo — run the causal spine on real bridge-collapse prose through the
//! actual ingest wiring (`mint_causal_spine_edges`) and dump the causal graph it
//! produces: event nodes + typed causal edges (OpenIE entity→entity + CATENA
//! event→event). Requires the spaCy model; skips without it.
//!
//! Run:  SHODH_SPACY_MODEL_PATH=<en_core_web_sm bundle> \
//!         cargo test --test causal_spine_live -- --nocapture

use chrono::Utc;
use uuid::Uuid;

use shodh_memory::dep_parser;
use shodh_memory::graph_memory::{EntityLabel, EntityNode, GraphMemory};

// Real Francis Scott Key Bridge collapse narrative (2024-03-26).
const PASSAGES: &[&str] = &[
    "The container ship Dali lost power and rammed the Francis Scott Key Bridge, causing the bridge to collapse into the Patapsco River.",
    "The collapse triggered a massive emergency response as rescue teams searched the water.",
    "Because of the collision, the Port of Baltimore was closed to ship traffic.",
    "The bridge collapsed after the ship struck one of its main support columns.",
    "The disaster killed six construction workers who were on the bridge.",
    "Officials later funded the rebuild of the bridge and restored the port.",
];

// Known participants (in production these come from GLiNER at ingest).
const ENTITIES: &[(&str, &str)] = &[
    ("Dali", "Product"),
    ("container ship", "Product"),
    ("Francis Scott Key Bridge", "Location"),
    ("bridge", "Location"),
    ("Patapsco River", "Location"),
    ("Port of Baltimore", "Location"),
    ("ship", "Product"),
    ("construction workers", "Person"),
    ("Officials", "Person"),
    ("port", "Location"),
];

fn label_of(s: &str) -> EntityLabel {
    match s {
        "Location" => EntityLabel::Location,
        "Person" => EntityLabel::Person,
        "Product" => EntityLabel::Product,
        _ => EntityLabel::Concept,
    }
}

#[test]
fn causal_spine_on_real_bridge_prose() {
    if !dep_parser::is_available() {
        eprintln!("SKIP causal_spine_on_real_bridge_prose: set SHODH_SPACY_MODEL_PATH");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let graph = GraphMemory::new(dir.path(), None).unwrap();

    // Pre-create the participant entity nodes.
    let mut entities: Vec<(String, Uuid, EntityLabel)> = Vec::new();
    for (name, ty) in ENTITIES {
        let label = label_of(ty);
        let node = EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![label.clone()],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 1.0,
            is_proper_noun: true,
            selectivity: None,
        };
        let uuid = graph.add_entity(node).unwrap();
        entities.push((name.to_string(), uuid, label));
    }

    // Ingest each passage through the actual causal-spine wiring.
    let mut minted = 0usize;
    for p in PASSAGES {
        minted += graph.mint_causal_spine_edges(p, &entities, Uuid::new_v4(), Utc::now());
    }

    // Dump the resulting causal graph.
    let all = graph.get_all_entities().unwrap();
    let name_of: std::collections::HashMap<Uuid, String> =
        all.iter().map(|e| (e.uuid, e.name.clone())).collect();
    let events: Vec<&EntityNode> = all
        .iter()
        .filter(|e| e.labels.contains(&EntityLabel::Event))
        .collect();
    let rels = graph.get_all_relationships().unwrap();

    println!("\n================ CAUSAL SPINE (real bridge prose) ================");
    println!("passages ingested: {}", PASSAGES.len());
    println!("causal-spine edges minted: {minted}");
    println!("\nEVENT nodes created ({}):", events.len());
    for e in &events {
        println!("   • {}", e.name);
    }
    println!("\nCAUSAL edges ({}):", rels.len());
    for r in &rels {
        let src = name_of.get(&r.from_entity).cloned().unwrap_or_default();
        let dst = name_of.get(&r.to_entity).cloned().unwrap_or_default();
        let by = r
            .provenance
            .first()
            .and_then(|p| p.typed_by)
            .map(|m| format!("{m:?}"))
            .unwrap_or_default();
        println!(
            "   {src}  --[{}]-->  {dst}   ({by})",
            r.relation_type.as_str()
        );
    }
    println!("=================================================================\n");

    assert!(
        minted > 0,
        "expected the spine to mint edges on causal prose"
    );
    assert!(
        !events.is_empty(),
        "expected CATENA event nodes (collapse, collision, …)"
    );
}
