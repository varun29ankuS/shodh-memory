//! Bundle loader: reads model.json (manifest) + model.safetensors (f32 weights).

use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub struct Bundle {
    pub manifest: Value,
    pub tensors: HashMap<String, Tensor>,
}

impl Bundle {
    pub fn load(dir: &Path) -> Bundle {
        let manifest = fs::read_to_string(dir.join("model.json")).unwrap();
        let buf = fs::read(dir.join("model.safetensors")).unwrap();
        Bundle::from_bytes(&manifest, &buf)
    }

    /// Load from in-memory bytes (browser/wasm: no filesystem).
    pub fn from_bytes(manifest_json: &str, safetensors: &[u8]) -> Bundle {
        let manifest: Value = serde_json::from_str(manifest_json).unwrap();
        let st = SafeTensors::deserialize(safetensors).unwrap();
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            assert_eq!(
                view.dtype(),
                safetensors::Dtype::F32,
                "tensor {} is not f32",
                name
            );
            let data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.insert(
                name.to_string(),
                Tensor { shape: view.shape().to_vec(), data },
            );
        }
        Bundle { manifest, tensors }
    }

    pub fn get(&self, key: &str) -> &Tensor {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor: {}", key))
    }

    pub fn symbols(&self) -> HashMap<String, u64> {
        let mut m = HashMap::new();
        if let Some(obj) = self.manifest["symbols"].as_object() {
            for (k, v) in obj {
                if let Some(n) = v.as_u64() {
                    m.insert(k.clone(), n);
                }
            }
        }
        m
    }
}
