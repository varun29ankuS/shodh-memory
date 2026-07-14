//! Hybrid Decay Model (SHO-103)
//!
//! Implements biologically-accurate memory decay based on neuroscience research.
//!
//! # The Problem with Pure Exponential Decay
//!
//! Traditional memory systems use exponential decay: `w(t) = w₀ × e^(-λt)`
//!
//! This produces a "cliff" effect where memories drop rapidly and then flatten:
//! - Day 1: 100% → 95%
//! - Day 7: 95% → 70%
//! - Day 30: 70% → 15% (steep cliff)
//!
//! # The Solution: Hybrid Decay
//!
//! Human memory follows a power-law for long-term retention, not exponential.
//!
//! This module implements a hybrid model:
//! - **Consolidation phase** (t < 3 days): Exponential decay
//!   - Fast filtering of noise and weak associations
//!   - Matches short-term synaptic depression
//! - **Long-term phase** (t ≥ 3 days): Power-law decay
//!   - Heavy tail preserves important memories longer
//!   - Matches empirical human forgetting curves
//!
//! ```text
//!         Exponential              Power-Law
//!         (consolidation)          (long-term retention)
//!
//! Strength │ ╲
//!     100% │  ╲
//!          │   ╲
//!      60% │    ╲___
//!          │        ╲____
//!      30% │             ╲________
//!          │                      ╲___________
//!       5% │─────────────────────────────────────────
//!          └────┬────────┬─────────────────────────► Time
//!               │        │
//!            t_cross   (days)
//!          (3 days)
//! ```
//!
//! # References
//!
//! - Wixted & Ebbesen (1991) "On the Form of Forgetting"
//! - Wixted (2004) "The psychology and neuroscience of forgetting"
//! - Anderson & Schooler (1991) "Reflections of the Environment in Memory"

use crate::constants::{
    DECAY_CROSSOVER_DAYS, DECAY_LAMBDA_CONSOLIDATION, POWERLAW_BETA, POWERLAW_BETA_POTENTIATED,
};

/// Calculates the hybrid decay factor for a given elapsed time.
///
/// Returns a value between 0.0 and 1.0 representing the retention ratio.
///
/// # Arguments
///
/// * `days_elapsed` - Time since last activation in days
/// * `potentiated` - Whether this is a potentiated/important memory (uses slower decay)
///
/// # Returns
///
/// Decay factor to multiply with original strength: `new_strength = old_strength * decay_factor`
///
/// # Example
///
/// ```ignore
/// let factor = hybrid_decay_factor(7.0, false);
/// let new_strength = old_strength * factor;
/// ```
#[inline]
pub fn hybrid_decay_factor(days_elapsed: f64, potentiated: bool) -> f32 {
    if days_elapsed <= 0.0 {
        return 1.0;
    }

    let beta = if potentiated {
        POWERLAW_BETA_POTENTIATED
    } else {
        POWERLAW_BETA
    };

    // Exponential rate for consolidation phase
    // Potentiated memories use slower exponential decay too
    let lambda = if potentiated {
        DECAY_LAMBDA_CONSOLIDATION * 0.5 // Half the rate for potentiated
    } else {
        DECAY_LAMBDA_CONSOLIDATION
    };

    if days_elapsed < DECAY_CROSSOVER_DAYS {
        // Consolidation phase: exponential decay
        // w(t) = w₀ × e^(-λt)
        (-lambda * days_elapsed).exp() as f32
    } else {
        // Long-term phase: power-law decay
        // First, calculate what value we'd have at crossover with exponential
        let value_at_crossover = (-lambda * DECAY_CROSSOVER_DAYS).exp();

        // Then apply power-law from crossover point
        // A(t) = A_cross × (t / t_cross)^(-β)
        let power_law_factor = (days_elapsed / DECAY_CROSSOVER_DAYS).powf(-beta);

        (value_at_crossover * power_law_factor) as f32
    }
}

/// Tier-aware decay factor for edge consolidation (3-tier memory model)
///
/// Each tier has different decay characteristics based on hippocampal-cortical research:
/// - L1 (Working): ~2.9%/hour decay (λ=0.029), max 48 hours
/// - L2 (Episodic): ~3.1%/day decay (λ=0.031), max 30 days
/// - L3 (Semantic): ~2%/month decay (λ=0.02/720h), near-permanent
///
/// # Arguments
///
/// * `hours_elapsed` - Time since last activation in hours
/// * `tier` - Memory tier (0=L1, 1=L2, 2=L3)
/// * `ltp_decay_factor` - LTP decay protection factor (1.0=none, 0.5=2x slower, 0.1=10x slower)
///
/// # Returns
///
/// Decay factor (0.0-1.0) and whether edge should be pruned
///
/// # PIPE-4 Update
///
/// Changed from `potentiated: bool` to `ltp_decay_factor: f32` to support
/// multi-scale LTP with graduated protection levels:
/// - LtpStatus::None → 1.0 (no protection)
/// - LtpStatus::Burst → 0.5 (2x slower decay, temporary)
/// - LtpStatus::Weekly → 0.3 (3x slower decay, moderate)
/// - LtpStatus::Full → 0.1 (10x slower decay, maximum)
#[inline]
pub fn tier_decay_factor(hours_elapsed: f64, tier: u8, ltp_decay_factor: f32) -> (f32, bool) {
    use crate::constants::*;

    if hours_elapsed <= 0.0 {
        return (1.0, false);
    }

    let (decay_rate, max_age_hours, prune_threshold) = match tier {
        0 => {
            // L1 Working: ~2.9%/hour decay (λ=0.029), max 48 hours
            (
                L1_DECAY_PER_HOUR as f64,
                (L1_MAX_AGE_HOURS as f64),
                L1_PRUNE_THRESHOLD,
            )
        }
        1 => {
            // L2 Episodic: ~3.1%/day decay (λ=0.031), max 30 days
            let decay_per_hour = L2_DECAY_PER_DAY as f64 / 24.0;
            (
                decay_per_hour,
                (L2_MAX_AGE_DAYS as f64) * 24.0,
                L2_PRUNE_THRESHOLD,
            )
        }
        _ => {
            // L3 Semantic (tier 2+): 2%/month decay, near-permanent
            let decay_per_hour = L3_DECAY_PER_MONTH as f64 / (30.0 * 24.0);
            // Max age: effectively unlimited (10 years)
            (decay_per_hour, 87600.0, L3_PRUNE_THRESHOLD)
        }
    };

    // PIPE-4: Apply graduated LTP protection
    // ltp_decay_factor of 0.5 = 2x slower, 0.1 = 10x slower, 1.0 = no protection
    let effective_rate = decay_rate * ltp_decay_factor as f64;

    // Exponential decay: w(t) = w₀ × e^(-λt)
    let decay_factor = (-effective_rate * hours_elapsed).exp() as f32;

    // Check if edge exceeded max age (should prune)
    // PIPE-4: Potentiated edges (ltp_decay_factor < 1.0) extend max age proportionally
    let effective_max_age = if ltp_decay_factor < 1.0 {
        max_age_hours / (ltp_decay_factor as f64).max(0.01)
    } else {
        max_age_hours
    };
    let should_prune = hours_elapsed > effective_max_age && decay_factor < prune_threshold;

    (decay_factor.max(0.001), should_prune)
}

// =============================================================================
// Topology-aware decay (W1-B) — bridge protection in the prune gate.
//
// A low-traffic node that is the ONLY connector between two clusters is exactly
// what multi-hop retrieval needs alive (measured: fragment bridges took lineage
// r@10 0.05 → 1.0). Time+usage decay is blind to that: an equally-old bridge and
// an equally-old dense-cluster leaf look identical to `RelationshipEdge::decay`.
//
// This module computes, in the consolidation "sleep" pass, a per-node structural
// criticality score (articulation points + bridge edges via one iterative
// Tarjan/lowlink DFS), smooths it across cycles (hysteresis, because articulation
// status flickers as edges churn), and exposes a prune-gate term that rescues a
// budgeted top slice of prune-candidate edges whose removal would fragment the
// graph. Everything here is gated behind `SHODH_TOPOLOGY_AWARE_DECAY` at the call
// site; with the flag off the prune decision is byte-identical to today's.
// =============================================================================

use std::collections::{HashMap, HashSet};

/// Output of the pure, index-based Tarjan pass over an undirected SIMPLE graph.
///
/// The caller collapses parallel/directed edges into a simple undirected graph
/// before calling [`tarjan_topology`], so bridge detection can use parent-node
/// tracking (a genuine parallel edge would otherwise mask a false bridge).
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleTopology {
    /// `articulation[i]` — removing node `i` increases the component count.
    pub articulation: Vec<bool>,
    /// Undirected bridge edges as `(min_index, max_index)` — removing the edge
    /// increases the component count.
    pub bridges: Vec<(usize, usize)>,
    /// Per-node structural criticality in `[0, 1]`: the largest graph partition a
    /// node induces (as articulation point or bridge endpoint), normalized by the
    /// maximum possible split within its connected component (`c²/4`). `0` for
    /// nodes that are neither articulation points nor bridge endpoints.
    pub node_score: Vec<f32>,
    /// Per-bridge normalized split score, aligned with `bridges`.
    pub bridge_score: Vec<f32>,
}

/// Compute articulation points AND bridge edges in a single iterative
/// Tarjan/lowlink DFS — `O(V + E)`, no recursion.
///
/// `adj` must be an undirected SIMPLE adjacency list (no self-loops, no parallel
/// edges, symmetric: `v ∈ adj[u] ⟺ u ∈ adj[v]`). Recursion is avoided on purpose:
/// a production entity graph can be tens of thousands of nodes deep along a chain,
/// and a recursive DFS would blow the stack. The explicit `stack` holds
/// `(node, parent, next_neighbor_cursor)` frames; the "on the way up" work
/// (low-link relaxation, subtree-size accumulation, articulation/bridge tests)
/// runs when a frame is popped, exactly mirroring the post-order step of the
/// recursive formulation.
///
/// # Correctness of the iterative stack
///
/// * Each node is pushed exactly once (guarded by `disc[v] < 0`), so the frame
///   count is bounded by `V` and the total neighbor-cursor advance by `2E`.
/// * The parent edge is skipped exactly once per frame (`skipped_parent`) so a
///   real second connection to the parent is still seen as a back edge — required
///   for correct low-link values.
/// * Split products need the whole component size, which is only known once the
///   component's root frame is popped; articulation/bridge magnitudes are
///   therefore recorded as `(node, child_subtree_size)` events during the DFS and
///   resolved against `comp_size` afterward.
pub fn tarjan_topology(adj: &[Vec<usize>]) -> SimpleTopology {
    let n = adj.len();
    let mut disc: Vec<i64> = vec![-1; n];
    let mut low: Vec<i64> = vec![0; n];
    let mut size: Vec<u64> = vec![1; n];
    let mut articulation = vec![false; n];
    let mut node_split: Vec<f64> = vec![0.0; n];
    let mut bridges: Vec<(usize, usize)> = Vec::new();
    let mut bridge_split_raw: Vec<f32> = Vec::new();

    let mut timer: i64 = 0;

    // Per-frame DFS state: (node, parent, cursor, skipped_parent_edge_once).
    struct Frame {
        node: usize,
        parent: i64,
        cursor: usize,
        skipped_parent: bool,
    }

    for s in 0..n {
        if disc[s] >= 0 {
            continue;
        }

        // --- One connected component rooted at `s` ---------------------------
        // Deferred articulation events: (cut_node, detached_subtree_size). The
        // split product is finished once the component size is known.
        let mut artic_events: Vec<(usize, u64)> = Vec::new();
        // Bridge events: (parent, child, child_subtree_size).
        let mut bridge_events: Vec<(usize, usize, u64)> = Vec::new();
        let mut root_child_events: Vec<u64> = Vec::new();
        let mut root_children = 0usize;

        disc[s] = timer;
        low[s] = timer;
        timer += 1;
        let mut stack: Vec<Frame> = vec![Frame {
            node: s,
            parent: -1,
            cursor: 0,
            skipped_parent: false,
        }];

        while let Some(frame) = stack.last_mut() {
            let u = frame.node;
            if frame.cursor < adj[u].len() {
                let v = adj[u][frame.cursor];
                frame.cursor += 1;

                // Skip the single parent edge once (simple graph ⇒ at most one).
                if !frame.skipped_parent && frame.parent >= 0 && v == frame.parent as usize {
                    frame.skipped_parent = true;
                    continue;
                }
                if v == u {
                    continue; // defensive: caller strips self-loops
                }

                if disc[v] < 0 {
                    disc[v] = timer;
                    low[v] = timer;
                    timer += 1;
                    if u == s {
                        root_children += 1;
                    }
                    stack.push(Frame {
                        node: v,
                        parent: u as i64,
                        cursor: 0,
                        skipped_parent: false,
                    });
                } else {
                    // Back edge (or forward/cross in undirected DFS = back edge).
                    if disc[v] < low[u] {
                        low[u] = disc[v];
                    }
                }
            } else {
                // Post-order: `u` is finished. Relax parent and emit events.
                let low_u = low[u];
                let disc_u = disc[u];
                let size_u = size[u];
                let parent = frame.parent;
                stack.pop();

                if parent >= 0 {
                    let p = parent as usize;
                    if low_u < low[p] {
                        low[p] = low_u;
                    }
                    size[p] += size_u;

                    if low_u > disc[p] {
                        // Removing edge (p, u) disconnects `u`'s subtree ⇒ bridge.
                        bridge_events.push((p, u, size_u));
                    }
                    if p == s {
                        // Root articulation is decided by child count, recorded here.
                        root_child_events.push(size_u);
                    } else if low_u >= disc[p] {
                        // Non-root cut vertex: removing `p` detaches `u`'s subtree.
                        articulation[p] = true;
                        artic_events.push((p, size_u));
                    }
                }
                let _ = disc_u; // (kept for readability of the post-order step)
            }
        }

        // Root is an articulation point iff it has ≥2 DFS-tree children.
        if root_children >= 2 {
            articulation[s] = true;
            for &sz in &root_child_events {
                artic_events.push((s, sz));
            }
        }

        // --- Resolve split magnitudes now that comp_size is known ------------
        let comp_size = size[s];
        let denom = (comp_size as f64 * comp_size as f64) / 4.0;
        let denom = if denom > 0.0 { denom } else { 1.0 };

        for (p, u, size_u) in bridge_events {
            let split = (size_u as f64) * ((comp_size - size_u) as f64);
            if split > node_split[p] {
                node_split[p] = split;
            }
            if split > node_split[u] {
                node_split[u] = split;
            }
            let (a, b) = if p < u { (p, u) } else { (u, p) };
            bridges.push((a, b));
            bridge_split_raw.push((split / denom) as f32);
        }
        for (p, size_u) in artic_events {
            let split = (size_u as f64) * ((comp_size - size_u) as f64);
            if split > node_split[p] {
                node_split[p] = split;
            }
        }
    }

    // node_split holds RAW split products across components of different sizes,
    // so normalize each node against its own component's c²/4 (same denominator
    // the bridge scores use). Component membership/size is re-derived inside
    // `normalize_node_scores` by an iterative flood fill over `adj` — the DFS
    // `size` array is passed through but intentionally unused there (flood fill
    // is the single source of truth; reusing DFS root sizes is a known
    // constant-factor optimization if this ever runs default-ON).
    let node_score = normalize_node_scores(adj, &disc, &size, &node_split);

    SimpleTopology {
        articulation,
        bridges,
        node_score,
        bridge_score: bridge_split_raw,
    }
}

/// Normalize each node's raw split product against its connected component's
/// theoretical maximum (`c²/4`, where `c` is the component size). Component
/// membership is recovered by a linear scan that groups nodes sharing a DFS root
/// — cheaper than storing roots inline and keeps [`tarjan_topology`] readable.
fn normalize_node_scores(
    adj: &[Vec<usize>],
    disc: &[i64],
    size: &[u64],
    node_split: &[f64],
) -> Vec<f32> {
    let n = adj.len();
    let mut comp_size = vec![0u64; n];

    // Recover components with a cheap iterative flood fill over the simple graph.
    let mut seen = vec![false; n];
    for s in 0..n {
        if seen[s] || disc[s] < 0 {
            continue;
        }
        let mut members = Vec::new();
        let mut stack = vec![s];
        seen[s] = true;
        while let Some(u) = stack.pop() {
            members.push(u);
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        let c = members.len() as u64;
        for u in members {
            comp_size[u] = c;
        }
    }
    let _ = size; // size at roots equals comp_size; flood fill is the source of truth.

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        if node_split[i] <= 0.0 || comp_size[i] < 2 {
            continue;
        }
        let c = comp_size[i] as f64;
        let denom = (c * c) / 4.0;
        let v = (node_split[i] / denom).clamp(0.0, 1.0);
        out[i] = v as f32;
    }
    out
}

/// Per-cycle topology protection derived from the entity graph, keyed by entity
/// UUID. Held on the graph and smoothed across cycles by [`smooth_protection`].
#[derive(Debug, Clone, Default)]
pub struct TopologyProtection {
    /// Smoothed per-node protection in `[0, 1]`.
    pub node_protection: HashMap<uuid::Uuid, f32>,
    /// Undirected endpoint pairs `(min, max)` that are bridges THIS cycle.
    pub bridge_pairs: HashSet<(uuid::Uuid, uuid::Uuid)>,
}

impl TopologyProtection {
    /// Prune-gate protection for an edge `(from, to)`.
    ///
    /// A bridge edge is the direct single-point-of-failure between two clusters,
    /// so it inherits the STRONGER endpoint's protection (`max`). A non-bridge
    /// edge is only protected when BOTH endpoints are structurally critical
    /// (`min`) — this is what stops an articulation hub's redundant intra-cluster
    /// edges (hub critical, cluster-mate not) from being protected, while still
    /// protecting a chain of cut vertices where every link matters.
    pub fn edge_protection(&self, from: &uuid::Uuid, to: &uuid::Uuid) -> f32 {
        let pf = self.node_protection.get(from).copied().unwrap_or(0.0);
        let pt = self.node_protection.get(to).copied().unwrap_or(0.0);
        let (a, b) = if from <= to {
            (*from, *to)
        } else {
            (*to, *from)
        };
        if self.bridge_pairs.contains(&(a, b)) {
            pf.max(pt)
        } else {
            pf.min(pt)
        }
    }
}

/// Run the topology pass over a set of directed graph edges (collapsed to an
/// undirected simple graph) and return the RAW (un-smoothed) per-node protection
/// plus the current bridge-pair set. Isolated self-loops and parallel/directed
/// duplicates are collapsed.
pub fn compute_topology_protection(edges: &[(uuid::Uuid, uuid::Uuid)]) -> TopologyProtection {
    // Index the distinct entities that appear in an active edge.
    let mut index: HashMap<uuid::Uuid, usize> = HashMap::new();
    let mut ids: Vec<uuid::Uuid> = Vec::new();
    let idx = |id: uuid::Uuid,
               index: &mut HashMap<uuid::Uuid, usize>,
               ids: &mut Vec<uuid::Uuid>|
     -> usize {
        if let Some(&i) = index.get(&id) {
            i
        } else {
            let i = ids.len();
            index.insert(id, i);
            ids.push(id);
            i
        }
    };

    // Deduped undirected adjacency as neighbor sets, then materialized to Vecs.
    let mut nbr: Vec<HashSet<usize>> = Vec::new();
    for &(a, b) in edges {
        if a == b {
            continue; // no self-loops in the structural graph
        }
        let ia = idx(a, &mut index, &mut ids);
        let ib = idx(b, &mut index, &mut ids);
        while nbr.len() <= ia.max(ib) {
            nbr.push(HashSet::new());
        }
        nbr[ia].insert(ib);
        nbr[ib].insert(ia);
    }
    let adj: Vec<Vec<usize>> = nbr.iter().map(|s| s.iter().copied().collect()).collect();

    let topo = tarjan_topology(&adj);

    let mut node_protection = HashMap::with_capacity(ids.len());
    for (i, id) in ids.iter().enumerate() {
        let s = topo.node_score.get(i).copied().unwrap_or(0.0);
        if s > 0.0 {
            node_protection.insert(*id, s);
        }
    }
    let mut bridge_pairs = HashSet::with_capacity(topo.bridges.len());
    for &(a, b) in &topo.bridges {
        let ua = ids[a];
        let ub = ids[b];
        let (x, y) = if ua <= ub { (ua, ub) } else { (ub, ua) };
        bridge_pairs.insert((x, y));
    }

    TopologyProtection {
        node_protection,
        bridge_pairs,
    }
}

/// Hysteresis: smooth this cycle's raw protection against the previous cycle's
/// smoothed map. `protection = max(new_score, old · decay)`.
///
/// Articulation status flickers as edges churn: a single edge added or pruned can
/// flip a node in and out of the cut-vertex set between cycles. Toggling
/// protection with it would defeat the point — a bridge that is briefly bypassed
/// (then reinstated) must not be forgotten in the one cycle it is redundant. The
/// smoothed value decays geometrically by `decay` (default 0.5) each cycle a node
/// is NOT critical, so protection persists ≈4 cycles (1.0 → 0.5 → 0.25 → 0.125,
/// below the rescue floor by cycle 3–4). At the production ~6h heavy-cycle
/// cadence that is ≈1 day of structural memory — long enough to ride out edge
/// churn, short enough that a genuinely dissolved bridge stops being protected
/// within a day.
pub fn smooth_protection(
    old: &HashMap<uuid::Uuid, f32>,
    new_raw: &HashMap<uuid::Uuid, f32>,
    decay: f32,
) -> HashMap<uuid::Uuid, f32> {
    let mut out: HashMap<uuid::Uuid, f32> = HashMap::with_capacity(new_raw.len().max(old.len()));
    // Carry decayed old protection forward (nodes no longer critical this cycle).
    for (id, &v) in old {
        let decayed = v * decay;
        if decayed > 1e-3 {
            out.insert(*id, decayed);
        }
    }
    // Raw score this cycle wins if higher.
    for (id, &v) in new_raw {
        let e = out.entry(*id).or_insert(0.0);
        if v > *e {
            *e = v;
        }
    }
    out
}

/// Prune-gate keep score: `keep = usage_term + α · bridge_protection`.
///
/// `usage_term` is the edge's own post-decay strength (the signal the base gate
/// already used to decide "prune"). Topology protection lifts that effective
/// strength; the caller rescues the edge iff `keep_score` clears the tier prune
/// threshold AND the edge is within the per-cycle rescue budget.
#[inline]
pub fn topology_keep_score(usage_term: f32, bridge_protection: f32, alpha: f32) -> f32 {
    usage_term + alpha * bridge_protection
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decay_at_zero() {
        assert_eq!(hybrid_decay_factor(0.0, false), 1.0);
        assert_eq!(hybrid_decay_factor(-1.0, false), 1.0);
    }

    #[test]
    fn test_exponential_phase() {
        // During consolidation (< 3 days), should be exponential
        let factor_1day = hybrid_decay_factor(1.0, false);
        let factor_2day = hybrid_decay_factor(2.0, false);

        // Exponential property: ratio should be constant
        let ratio_1_to_2 = factor_2day / factor_1day;
        let expected_ratio = (-DECAY_LAMBDA_CONSOLIDATION).exp() as f32;

        assert!((ratio_1_to_2 - expected_ratio).abs() < 0.01);
    }

    #[test]
    fn test_powerlaw_phase() {
        // After crossover (> 3 days), should be power-law
        let factor_7day = hybrid_decay_factor(7.0, false);
        let factor_14day = hybrid_decay_factor(14.0, false);

        // Power-law property: doubling time should give 2^(-β) ratio
        let ratio = factor_14day / factor_7day;
        let expected_ratio = 2.0_f64.powf(-POWERLAW_BETA) as f32;

        assert!((ratio - expected_ratio).abs() < 0.02);
    }

    #[test]
    fn test_continuity_at_crossover() {
        // Values just before and after crossover should be close
        let just_before = hybrid_decay_factor(DECAY_CROSSOVER_DAYS - 0.001, false);
        let just_after = hybrid_decay_factor(DECAY_CROSSOVER_DAYS + 0.001, false);

        assert!((just_before - just_after).abs() < 0.01);
    }

    #[test]
    fn test_potentiated_decays_slower() {
        let normal = hybrid_decay_factor(30.0, false);
        let potentiated = hybrid_decay_factor(30.0, true);

        // Potentiated should retain more
        assert!(potentiated > normal);
    }

    #[test]
    fn test_heavy_tail_retention() {
        // Key property: power-law has heavy tail
        // At 365 days, we should still have meaningful retention
        let year_retention = hybrid_decay_factor(365.0, false);
        let year_retention_potentiated = hybrid_decay_factor(365.0, true);

        // Normal: should be > 1%
        assert!(year_retention > 0.01);
        // Potentiated: should be > 5%
        assert!(year_retention_potentiated > 0.05);
    }

    #[test]
    fn test_tier_decay_factor_l1_with_and_without_ltp() {
        let (unprotected, _) = tier_decay_factor(24.0, 0, 1.0);
        let (protected, _) = tier_decay_factor(24.0, 0, 0.5);
        assert!(protected > unprotected);
    }

    #[test]
    fn test_tier_decay_factor_l1_prune_threshold() {
        let (factor_at_max_age, should_prune_at_max_age) = tier_decay_factor(48.0, 0, 1.0);
        // At max age boundary, L1 should still not prune because pruning requires "greater than" max age.
        assert!(factor_at_max_age > 0.1);
        assert!(!should_prune_at_max_age);

        let (factor_past_max_age, should_prune_past_max_age) = tier_decay_factor(96.0, 0, 1.0);
        assert!(factor_past_max_age < 0.1);
        assert!(should_prune_past_max_age);
    }

    #[test]
    fn test_tier_decay_factor_l3_long_tail() {
        let (factor_1y, prune_1y) = tier_decay_factor(365.0 * 24.0, 2, 1.0);
        assert!(factor_1y > 0.7);
        assert!(!prune_1y);

        let (factor_3y, prune_3y) = tier_decay_factor(3.0 * 365.0 * 24.0, 2, 1.0);
        assert!(factor_3y > 0.45);
        assert!(!prune_3y);
    }

    #[test]
    fn test_tier_decay_zero_and_negative_elapsed() {
        let (zero_factor, zero_prune) = tier_decay_factor(0.0, 1, 1.0);
        assert_eq!(zero_factor, 1.0);
        assert!(!zero_prune);

        let (neg_factor, neg_prune) = tier_decay_factor(-10.0, 1, 1.0);
        assert_eq!(neg_factor, 1.0);
        assert!(!neg_prune);
    }

    #[test]
    fn test_tier_decay_invalid_tier_defaults_to_l3() {
        let (invalid_tier, _) = tier_decay_factor(24.0, 9, 1.0);
        let (l3, _) = tier_decay_factor(24.0, 2, 1.0);
        assert_eq!(invalid_tier, l3);
    }

    // =========================================================================
    // Topology (iterative Tarjan) — exact articulation / bridge sets on known
    // small graphs. These pin the O(V+E) lowlink pass against textbook answers.
    // =========================================================================

    /// Build a symmetric undirected simple adjacency list from an edge list.
    fn undirected(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut nbr: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for &(a, b) in edges {
            nbr[a].insert(b);
            nbr[b].insert(a);
        }
        nbr.into_iter().map(|s| s.into_iter().collect()).collect()
    }

    fn artic_set(t: &SimpleTopology) -> HashSet<usize> {
        t.articulation
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect()
    }

    fn bridge_set(t: &SimpleTopology) -> HashSet<(usize, usize)> {
        t.bridges.iter().copied().collect()
    }

    #[test]
    fn tarjan_chain_all_interior_articulation_all_edges_bridges() {
        // 0-1-2-3-4 : every interior node is a cut vertex, every edge a bridge.
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let t = tarjan_topology(&adj);
        assert_eq!(artic_set(&t), HashSet::from([1, 2, 3]));
        assert_eq!(
            bridge_set(&t),
            HashSet::from([(0, 1), (1, 2), (2, 3), (3, 4)])
        );
        // Middle node splits 5 into 2|3 → product 6, /(25/4=6.25) ≈ 0.96 (max).
        let mid = t.node_score[2];
        assert!(mid > 0.9, "middle of chain most critical, got {mid}");
    }

    #[test]
    fn tarjan_cycle_no_articulation_no_bridges() {
        // 0-1-2-3-0 : a single cycle is 2-edge-connected — nothing critical.
        let adj = undirected(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let t = tarjan_topology(&adj);
        assert!(artic_set(&t).is_empty(), "cycle has no cut vertices");
        assert!(bridge_set(&t).is_empty(), "cycle has no bridges");
        assert!(t.node_score.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn tarjan_two_clusters_one_bridge() {
        // Two triangles {0,1,2} and {3,4,5} joined by the single edge (2,3).
        // Only (2,3) is a bridge; only 2 and 3 are cut vertices.
        let adj = undirected(
            6,
            &[
                (0, 1),
                (1, 2),
                (2, 0), // triangle A
                (3, 4),
                (4, 5),
                (5, 3), // triangle B
                (2, 3), // the bridge
            ],
        );
        let t = tarjan_topology(&adj);
        assert_eq!(artic_set(&t), HashSet::from([2, 3]));
        assert_eq!(bridge_set(&t), HashSet::from([(2, 3)]));
        // Bridge splits 6 into 3|3 → product 9, /(36/4=9) = 1.0.
        let bi = t.bridges.iter().position(|&e| e == (2, 3)).unwrap();
        assert!((t.bridge_score[bi] - 1.0).abs() < 1e-6);
        assert!(t.node_score[2] > 0.9 && t.node_score[3] > 0.9);
        // Non-connector triangle members are NOT critical.
        assert_eq!(t.node_score[0], 0.0);
        assert_eq!(t.node_score[5], 0.0);
    }

    #[test]
    fn tarjan_clique_no_articulation_no_bridges() {
        // K4 : fully redundant, nothing structural.
        let adj = undirected(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        let t = tarjan_topology(&adj);
        assert!(artic_set(&t).is_empty());
        assert!(bridge_set(&t).is_empty());
    }

    #[test]
    fn tarjan_star_center_is_the_only_articulation() {
        // Star: center 0 with pendant leaves 1..4. Every spoke is a bridge; only
        // the center is a cut vertex.
        let adj = undirected(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let t = tarjan_topology(&adj);
        assert_eq!(artic_set(&t), HashSet::from([0]));
        assert_eq!(
            bridge_set(&t),
            HashSet::from([(0, 1), (0, 2), (0, 3), (0, 4)])
        );
    }

    #[test]
    fn tarjan_disconnected_components_scored_independently() {
        // Two separate edges: 0-1 and 2-3. Each is its own component; each edge
        // is a bridge splitting its 2-node component 1|1 → normalized 1.0.
        let adj = undirected(4, &[(0, 1), (2, 3)]);
        let t = tarjan_topology(&adj);
        assert_eq!(bridge_set(&t), HashSet::from([(0, 1), (2, 3)]));
        assert!(
            artic_set(&t).is_empty(),
            "leaf endpoints are not cut vertices"
        );
        for &s in &t.bridge_score {
            assert!((s - 1.0).abs() < 1e-6, "1|1 split normalizes to 1.0");
        }
    }

    #[test]
    fn tarjan_empty_and_singleton_are_inert() {
        assert!(tarjan_topology(&[]).bridges.is_empty());
        let t = tarjan_topology(&[vec![]]);
        assert!(t.bridges.is_empty());
        assert!(t.articulation.iter().all(|&b| !b));
        assert_eq!(t.node_score, vec![0.0]);
    }

    #[test]
    fn tarjan_deep_chain_does_not_overflow_stack() {
        // 50k-node chain: recursion would blow the stack; the iterative pass must
        // complete and mark every interior node as an articulation point.
        let n = 50_000;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let adj = undirected(n, &edges);
        let t = tarjan_topology(&adj);
        assert_eq!(t.bridges.len(), n - 1, "every chain edge is a bridge");
        // Interior nodes (all but the two endpoints) are cut vertices.
        let cuts = t.articulation.iter().filter(|&&b| b).count();
        assert_eq!(cuts, n - 2);
    }

    #[test]
    fn compute_topology_protection_maps_uuids_and_bridge_pairs() {
        use uuid::Uuid;
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let c = Uuid::from_u128(3);
        let d = Uuid::from_u128(4);
        // Two triangles A{a,b,x} / B{c,d,y}? keep it minimal: a-b-c chain where b
        // is the cut vertex and both edges are bridges.
        let edges = vec![(a, b), (b, c), (c, d)];
        let prot = compute_topology_protection(&edges);
        let pa = prot.node_protection.get(&a).copied().unwrap_or(0.0);
        let pb = prot.node_protection.get(&b).copied().unwrap_or(0.0);
        let pc = prot.node_protection.get(&c).copied().unwrap_or(0.0);
        // b and c are interior of the a-b-c-d chain ⇒ cut vertices, most critical
        // (the middle edge splits the chain 2|2). Leaves a, d are bridge endpoints
        // of a lower-split bridge (1|3), so they carry SOME protection but strictly
        // less than the interior cut vertices — the continuous signal, not a flag.
        assert!(pb > 0.0 && pc > 0.0);
        assert!(pa > 0.0 && pa < pb, "leaf < interior: pa={pa} pb={pb}");
        // Bridge pairs are stored order-normalized.
        let pair_ab = if a <= b { (a, b) } else { (b, a) };
        assert!(prot.bridge_pairs.contains(&pair_ab));
        // A bridge edge takes the MAX of endpoint protection; a hub's redundant
        // edge would take MIN. Here every edge is a bridge.
        assert!(prot.edge_protection(&a, &b) > 0.0);
    }

    #[test]
    fn smooth_protection_decays_over_cycles_not_instantly() {
        use uuid::Uuid;
        let node = Uuid::from_u128(7);
        // Cycle 0: node is a strong bridge (raw 1.0).
        let mut old: HashMap<Uuid, f32> = HashMap::new();
        let mut raw: HashMap<Uuid, f32> = HashMap::new();
        raw.insert(node, 1.0);
        old = smooth_protection(&old, &raw, 0.5);
        assert!((old[&node] - 1.0).abs() < 1e-6);
        // Subsequent cycles: node is NO LONGER a bridge (raw empty). Protection
        // must decay geometrically, not vanish.
        let empty: HashMap<Uuid, f32> = HashMap::new();
        old = smooth_protection(&old, &empty, 0.5);
        assert!((old[&node] - 0.5).abs() < 1e-6, "cycle 1 → 0.5");
        old = smooth_protection(&old, &empty, 0.5);
        assert!((old[&node] - 0.25).abs() < 1e-6, "cycle 2 → 0.25");
        // Eventually falls below the tracking floor and is dropped.
        for _ in 0..10 {
            old = smooth_protection(&old, &empty, 0.5);
        }
        assert!(
            old.get(&node).is_none(),
            "protection fully released after enough quiet cycles"
        );
    }

    #[test]
    fn topology_keep_score_lifts_effective_strength() {
        // A near-floor bridge edge (strength 0.01) with full protection clears an
        // L2 prune threshold (0.2) under α=0.6; an unprotected one does not.
        let alpha = 0.6;
        assert!(topology_keep_score(0.01, 1.0, alpha) > 0.2);
        assert!(topology_keep_score(0.01, 0.0, alpha) < 0.2);
    }
}
