use std::collections::HashMap;
use std::sync::LazyLock;

use crate::graph_memory::RelationType;

pub static VERB_DICTIONARY: LazyLock<HashMap<&'static str, RelationType>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    let comp = ["infect", "compromise", "breach", "exploit"];
    for v in comp { m.insert(v, RelationType::Compromised); }
    
    let red = ["redirect", "forward", "proxy"];
    for v in red { m.insert(v, RelationType::RedirectsTo); }
    
    let host = ["host", "serve", "run"];
    for v in host { m.insert(v, RelationType::Hosts); }
    
    let cont = ["contain", "include", "embed"];
    for v in cont { m.insert(v, RelationType::ContainsExtraction); }
    
    let block = ["block", "filter", "deny"];
    for v in block { m.insert(v, RelationType::Blocks); }
    
    m
});

pub fn get_canonical_relation(verb: &str) -> Option<RelationType> {
    VERB_DICTIONARY.get(verb.to_lowercase().as_str()).cloned()
}
