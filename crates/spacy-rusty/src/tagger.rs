//! POS tagger: a single affine layer over the shared tok2vec output, argmax
//! over the fine-grained TAG labels (softmax is unnormalized at inference).

use crate::ml::{affine, argmax};
use crate::model::Bundle;

pub struct Tagger {
    w: Vec<f32>,
    b: Vec<f32>,
    n_o: usize,
    width: usize,
    pub labels: Vec<String>,
}

impl Tagger {
    pub fn load(b: &Bundle) -> Tagger {
        let cfg = &b.manifest["tagger"];
        let labels: Vec<String> =
            cfg["labels"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        let n_o = cfg["nO"].as_u64().unwrap() as usize;
        let wt = b.get("tagger.softmax.W");
        let width = wt.shape[1];
        Tagger { w: wt.data.clone(), b: b.get("tagger.softmax.b").data.clone(), n_o, width, labels }
    }

    /// Predict a fine-grained TAG label per token from its tok2vec vector.
    pub fn predict(&self, vecs: &[Vec<f32>]) -> Vec<String> {
        vecs.iter()
            .map(|v| {
                let scores = affine(v, &self.w, &self.b, self.n_o, self.width);
                self.labels[argmax(&scores)].clone()
            })
            .collect()
    }
}
