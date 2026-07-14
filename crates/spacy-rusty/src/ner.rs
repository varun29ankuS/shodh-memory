//! Transition-based NER (BiluoPushDown), greedy decode.
//!
//! Pipeline: NER's own 4-attr tok2vec -> reduce(96->64) -> PrecomputableAffine
//! lower (nF=3, nP=2) -> maxout -> upper linear (->74 moves). State features
//! are B(0), E(0) (open-entity start or -1), B(0)-1 (prev, or -1). Each step:
//! gather the 3 precomputed feature blocks (pad row for -1), sum + bias,
//! maxout over pieces, upper-score, mask invalid moves, argmax, transition.

use crate::model::Bundle;
use crate::tok2vec::Tok2Vec;
use crate::transition::Scorer;
use crate::vectors::Vectors;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Action {
    Begin,
    In,
    Last,
    Unit,
    Out,
}

pub struct Ner {
    tok2vec: Tok2Vec,
    scorer: Scorer,
    moves: Vec<(Action, Option<String>)>,
}

fn parse_move(name: &str) -> (Action, Option<String>) {
    if name == "O" {
        return (Action::Out, None);
    }
    let (pfx, rest) = name.split_at(1);
    let label = rest.strip_prefix('-').unwrap_or(rest);
    let action = match pfx {
        "B" => Action::Begin,
        "I" => Action::In,
        "L" => Action::Last,
        "U" => Action::Unit,
        _ => Action::Out,
    };
    let lbl = if label.is_empty() { None } else { Some(label.to_string()) };
    (action, lbl)
}

impl Ner {
    pub fn load(b: &Bundle, vectors: Option<Arc<Vectors>>) -> Ner {
        let cfg = &b.manifest["ner"];
        let t2v_cfg = cfg["tok2vec"].clone();
        let uses_static = t2v_cfg["include_static_vectors"].as_bool().unwrap_or(false);
        let tok2vec = Tok2Vec::load(b, &t2v_cfg, "ner.tok2vec", if uses_static { vectors } else { None });
        let scorer = Scorer::load(b, "ner");
        let moves: Vec<(Action, Option<String>)> = cfg["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| parse_move(m.as_str().unwrap()))
            .collect();
        Ner { tok2vec, scorer, moves }
    }

    /// Predict entity token spans (start, end_exclusive, label).
    /// `feats4[t]` are the 4 NER hash keys; `is_space[t]` flags whitespace tokens.
    pub fn predict(
        &self,
        feats4: &[Vec<u64>],
        orths: Option<&[u64]>,
        is_space: &[bool],
    ) -> Vec<(usize, usize, String)> {
        let n = feats4.len();
        if n == 0 {
            return vec![];
        }
        let tokvecs = self.tok2vec.forward(feats4, orths);
        let yf = self.scorer.precompute(&tokvecs);

        // greedy decode
        let mut i = 0usize;
        let mut open = false;
        let mut ent_start = 0usize;
        let mut ent_label = String::new();
        let mut ents: Vec<(usize, usize, String)> = vec![];

        while i < n {
            let e0: i64 = if open { ent_start as i64 } else { -1 };
            let ids: [i64; 3] = [
                i as i64,
                e0,
                if e0 == -1 { -1 } else { i as i64 - 1 },
            ];
            let scores = self.scorer.score(&yf, &ids);

            // pick argmax over valid moves
            let remaining = n - i;
            let cur_space = is_space[i];
            let mut best_c: isize = -1;
            let mut best_s = f32::NEG_INFINITY;
            for (c, (action, label)) in self.moves.iter().enumerate() {
                let valid = match action {
                    Action::Begin => !open && remaining >= 2 && label.is_some() && !cur_space,
                    Action::In => {
                        open && remaining >= 2 && label.as_deref() == Some(ent_label.as_str())
                    }
                    Action::Last => {
                        open && label.as_deref() == Some(ent_label.as_str())
                    }
                    Action::Unit => !open && label.is_some() && !cur_space,
                    Action::Out => !open,
                };
                if valid && scores[c] > best_s {
                    best_s = scores[c];
                    best_c = c as isize;
                }
            }
            if best_c < 0 {
                // no valid move (shouldn't happen) — treat as OUT
                i += 1;
                continue;
            }
            let (action, label) = &self.moves[best_c as usize];
            match action {
                Action::Begin => {
                    open = true;
                    ent_start = i;
                    ent_label = label.clone().unwrap();
                }
                Action::In => {}
                Action::Last => {
                    ents.push((ent_start, i + 1, ent_label.clone()));
                    open = false;
                }
                Action::Unit => {
                    ents.push((i, i + 1, label.clone().unwrap()));
                }
                Action::Out => {}
            }
            i += 1;
        }
        ents
    }
}
