//! Transition-based dependency parser (spaCy's arc-eager system), greedy decode.
//!
//! Unlike NER, the parser shares the MAIN tok2vec via a Tok2VecListener, so it
//! takes the already-computed 96-d token vectors and runs only reduce(96->64) ->
//! PrecomputableAffine lower (nF=8) -> maxout -> upper (-> n_moves). The 8 state
//! features are B(0),B(1),S(0),S(1),S(2),L(B(0),1),L(S(0),1),R(S(0),1). Moves
//! are S(shift) / D(reduce) / L-<dep>(left-arc) / R-<dep>(right-arc) / B(break).
//! Faithful port of spacy/pipeline/_parser_internals/{arc_eager,_state}.

use crate::model::Bundle;
use crate::transition::Scorer;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
enum PMove {
    Shift,
    Reduce,
    LeftArc(String),
    RightArc(String),
    Break,
}

fn parse_pmove(name: &str) -> PMove {
    if name == "S" {
        PMove::Shift
    } else if name == "D" {
        PMove::Reduce
    } else if let Some(l) = name.strip_prefix("L-") {
        PMove::LeftArc(l.to_string())
    } else if let Some(l) = name.strip_prefix("R-") {
        PMove::RightArc(l.to_string())
    } else {
        // "B" / "B-ROOT": Break ignores its label (it only marks a sent start).
        PMove::Break
    }
}

/// Per-token parse result. `head` is the head token index (== own index for a
/// root), `dep` the dependency label ("ROOT" for roots), `is_sent_start` true
/// for token 0 and every token a BREAK marked as a new sentence.
pub struct ParseOut {
    pub head: usize,
    pub dep: String,
    pub is_sent_start: bool,
}

pub struct Parser {
    scorer: Scorer,
    moves: Vec<PMove>,
}

impl Parser {
    /// Load the parser if present in the bundle (returns None otherwise).
    pub fn load(b: &Bundle) -> Option<Parser> {
        let present = b.manifest.get("parser").map(|v| !v.is_null()).unwrap_or(false);
        if !present {
            return None;
        }
        let scorer = Scorer::load(b, "parser");
        let moves: Vec<PMove> = b.manifest["parser"]["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| parse_pmove(m.as_str().unwrap()))
            .collect();
        Some(Parser { scorer, moves })
    }

    /// Parse a sequence, given the SHARED tok2vec output (one 96-d vec / token).
    pub fn predict(&self, vecs: &[Vec<f32>]) -> Vec<ParseOut> {
        let n = vecs.len();
        if n == 0 {
            return vec![];
        }
        let yf = self.scorer.precompute(vecs);
        let mut st = State::new(n);
        let max_iter = 3 * n + 5; // arc-eager terminates in <=2n (+breaks); guard anyway
        let mut it = 0;
        while !st.is_final() {
            it += 1;
            if it > max_iter {
                break;
            }
            let ids = st.context8();
            let scores = self.scorer.score(&yf, &ids);
            // argmax over grammar-valid moves
            let mut best_c: isize = -1;
            let mut best_s = f32::NEG_INFINITY;
            for (c, mv) in self.moves.iter().enumerate() {
                if st.is_valid(mv) && scores[c] > best_s {
                    best_s = scores[c];
                    best_c = c as isize;
                }
            }
            if best_c < 0 {
                break; // no valid move (shouldn't happen before final)
            }
            st.apply(&self.moves[best_c as usize]);
        }
        // Sentence starts come from the parse TREE, not the BREAK transition
        // (spaCy's set_children_from_heads): each root marks the leftmost token
        // of its subtree (l_edge) as a sentence start; token 0 is always one.
        // BREAK still matters during decode — it gates move validity via
        // is_sent_start — but the final flag is derived here.
        let heads_idx: Vec<usize> =
            (0..n).map(|i| if st.heads[i] < 0 { i } else { st.heads[i] as usize }).collect();
        let mut l_edge: Vec<usize> = (0..n).collect();
        let mut guard = 0;
        loop {
            // Relax l_edge[head] = min(l_edge over subtree); repeat until stable
            // (handles non-projective parses, as spaCy does).
            let mut changed = false;
            for i in 0..n {
                let h = heads_idx[i];
                if l_edge[i] < l_edge[h] {
                    l_edge[h] = l_edge[i];
                    changed = true;
                }
            }
            guard += 1;
            if !changed || guard > n + 10 {
                break;
            }
        }
        let mut sent_start = vec![false; n];
        for i in 0..n {
            if heads_idx[i] == i {
                sent_start[l_edge[i]] = true; // root -> leftmost token of its subtree
            }
        }
        sent_start[0] = true; // first token is always a sentence start (issue #2869)

        (0..n)
            .map(|i| {
                let (head, dep) = if st.heads[i] < 0 {
                    (i, "ROOT".to_string()) // head offset 0 == self == root
                } else {
                    (st.heads[i] as usize, st.deps[i].clone())
                };
                ParseOut { head, dep, is_sent_start: sent_start[i] }
            })
            .collect()
    }
}

/// Arc-eager parser state (port of StateC). Buffer is `b_i..n` plus a LIFO
/// `rebuffer` of tokens pushed back by REDUCE-unshift; `stack` is a LIFO of
/// token indices; arcs are kept per-head (left = head>child) so L()/R() can
/// fetch the nth child.
struct State {
    n: usize,
    stack: Vec<usize>,
    rebuffer: Vec<usize>,
    b_i: usize,
    heads: Vec<i32>,            // -1 = no head yet
    deps: Vec<String>,         // dep label of the arc into each token
    left_arcs: HashMap<usize, Vec<i32>>,  // head -> children (-1 = tombstone)
    right_arcs: HashMap<usize, Vec<i32>>,
    sent_starts: HashSet<usize>,
    unshiftable: Vec<bool>,
}

impl State {
    fn new(n: usize) -> State {
        State {
            n,
            stack: Vec::new(),
            rebuffer: Vec::new(),
            b_i: 0,
            heads: vec![-1; n],
            deps: vec![String::new(); n],
            left_arcs: HashMap::new(),
            right_arcs: HashMap::new(),
            sent_starts: HashSet::new(),
            unshiftable: vec![false; n],
        }
    }

    fn s(&self, i: i64) -> i64 {
        if i < 0 {
            return -1;
        }
        let i = i as usize;
        if i >= self.stack.len() {
            -1
        } else {
            self.stack[self.stack.len() - 1 - i] as i64
        }
    }

    fn b(&self, i: i64) -> i64 {
        if i < 0 {
            return -1;
        }
        let i = i as usize;
        if i < self.rebuffer.len() {
            self.rebuffer[self.rebuffer.len() - 1 - i] as i64
        } else {
            let bi = self.b_i + (i - self.rebuffer.len());
            if bi >= self.n {
                -1
            } else {
                bi as i64
            }
        }
    }

    fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    fn buffer_length(&self) -> usize {
        (self.n - self.b_i) + self.rebuffer.len()
    }

    fn is_final(&self) -> bool {
        self.stack.is_empty() && self.buffer_length() == 0
    }

    fn has_head(&self, c: i64) -> bool {
        c >= 0 && self.heads[c as usize] >= 0
    }

    fn is_sent_start(&self, w: i64) -> bool {
        w >= 0 && (w as usize) < self.n && self.sent_starts.contains(&(w as usize))
    }

    // Our inputs never pre-set token.sent_start = -1, so a token can always
    // start a sentence (matches spaCy when the parser runs from scratch).
    fn cannot_sent_start(&self, _w: i64) -> bool {
        false
    }

    fn is_unshiftable(&self, item: i64) -> bool {
        item >= 0 && (item as usize) < self.unshiftable.len() && self.unshiftable[item as usize]
    }

    fn nth_child(map: &HashMap<usize, Vec<i32>>, head: i64, idx: usize) -> i64 {
        if idx < 1 || head < 0 {
            return -1;
        }
        if let Some(arcs) = map.get(&(head as usize)) {
            let mut count = 0;
            for &child in arcs.iter().rev() {
                if child != -1 {
                    count += 1;
                    if count == idx {
                        return child as i64;
                    }
                }
            }
        }
        -1
    }

    fn l(&self, head: i64, idx: usize) -> i64 {
        Self::nth_child(&self.left_arcs, head, idx)
    }

    fn r(&self, head: i64, idx: usize) -> i64 {
        Self::nth_child(&self.right_arcs, head, idx)
    }

    fn context8(&self) -> [i64; 8] {
        // offset is always 0 for a from-scratch single-doc parse, so the
        // spaCy `ids[i] += offset` step is a no-op.
        [
            self.b(0),
            self.b(1),
            self.s(0),
            self.s(1),
            self.s(2),
            self.l(self.b(0), 1),
            self.l(self.s(0), 1),
            self.r(self.s(0), 1),
        ]
    }

    fn push(&mut self) {
        let b0 = if let Some(x) = self.rebuffer.pop() {
            x
        } else {
            let x = self.b_i;
            self.b_i += 1;
            x
        };
        self.stack.push(b0);
    }

    fn pop(&mut self) {
        self.stack.pop();
    }

    fn unshift(&mut self) {
        let s0 = *self.stack.last().unwrap();
        self.unshiftable[s0] = true;
        self.rebuffer.push(s0);
        self.stack.pop();
    }

    fn set_reshiftable(&mut self, item: i64) {
        if item >= 0 && (item as usize) < self.unshiftable.len() {
            self.unshiftable[item as usize] = false;
        }
    }

    fn set_sent_start(&mut self, w: i64) {
        if w >= 0 {
            self.sent_starts.insert(w as usize);
        }
    }

    fn add_arc(&mut self, head: i64, child: i64, label: &str) {
        let c = child as usize;
        if self.heads[c] >= 0 {
            self.del_arc(self.heads[c], child);
        }
        if head > child {
            self.left_arcs.entry(head as usize).or_default().push(child as i32);
        } else {
            self.right_arcs.entry(head as usize).or_default().push(child as i32);
        }
        self.heads[c] = head as i32;
        self.deps[c] = label.to_string();
    }

    fn del_arc(&mut self, h: i32, c: i64) {
        let ci = c as i32;
        let map = if (h as i64) > c { &mut self.left_arcs } else { &mut self.right_arcs };
        if let Some(arcs) = map.get_mut(&(h as usize)) {
            if arcs.is_empty() {
                return;
            }
            if *arcs.last().unwrap() == ci {
                arcs.pop();
            } else {
                let scan = arcs.len() - 1; // spaCy scans [0, size-1), tombstoning the first match
                for i in 0..scan {
                    if arcs[i] == ci {
                        arcs[i] = -1;
                        break;
                    }
                }
            }
        }
    }

    fn is_valid(&self, mv: &PMove) -> bool {
        match mv {
            PMove::Shift => {
                if self.stack_depth() == 0 {
                    true
                } else if self.buffer_length() < 2 {
                    false
                } else if self.is_sent_start(self.b(0)) {
                    false
                } else {
                    !self.is_unshiftable(self.b(0))
                }
            }
            PMove::Reduce => {
                if self.stack_depth() == 0 {
                    false
                } else if self.buffer_length() == 0 {
                    true
                } else if self.stack_depth() == 1 && self.cannot_sent_start(self.b(0)) {
                    false
                } else {
                    true
                }
            }
            // SUBTOK_LABEL constraint omitted: en_core_web_sm/md have no subtok move.
            PMove::LeftArc(_) | PMove::RightArc(_) => {
                self.stack_depth() > 0 && self.buffer_length() > 0 && !self.is_sent_start(self.b(0))
            }
            PMove::Break => {
                if self.buffer_length() < 2 {
                    false
                } else if self.b(1) != self.b(0) + 1 {
                    false
                } else if self.is_sent_start(self.b(1)) {
                    false
                } else {
                    !self.cannot_sent_start(self.b(1))
                }
            }
        }
    }

    fn apply(&mut self, mv: &PMove) {
        match mv {
            PMove::Shift => self.push(),
            PMove::Reduce => {
                if self.has_head(self.s(0)) || self.stack_depth() == 1 {
                    self.pop();
                } else {
                    self.unshift();
                }
            }
            PMove::LeftArc(label) => {
                let (b0, s0) = (self.b(0), self.s(0));
                self.add_arc(b0, s0, label);
                self.set_reshiftable(b0);
                self.pop();
            }
            PMove::RightArc(label) => {
                let (s0, b0) = (self.s(0), self.b(0));
                self.add_arc(s0, b0, label);
                self.push();
            }
            PMove::Break => {
                let b1 = self.b(1);
                self.set_sent_start(b1);
            }
        }
    }
}
