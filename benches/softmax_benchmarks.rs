//! Micro-benchmarks for argmax_softmax vs allocating softmax + max_by.
//!
//! Validates the perf claim that fused argmax_softmax avoids a per-token Vec
//! allocation on the NER hot path.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use shodh_memory::embeddings::ner::argmax_softmax;

/// Baseline: allocating softmax then argmax (the old code path).
fn softmax_then_argmax(logits: &[f32]) -> Option<(usize, f32)> {
    if logits.is_empty() {
        return None;
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp() / exp_sum).collect();
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, &prob)| (idx, prob))
}

fn bench_argmax_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmax_softmax");

    // NER label counts: tiny BERT has 9 labels (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)
    for size in [9, 16, 32, 128] {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 - 1.5).collect();

        group.bench_with_input(
            BenchmarkId::new("fused", size),
            &logits,
            |b, logits| {
                b.iter(|| argmax_softmax(black_box(logits)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("alloc_baseline", size),
            &logits,
            |b, logits| {
                b.iter(|| softmax_then_argmax(black_box(logits)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = softmax_benches;
    config = Criterion::default()
        .sample_size(200)
        .measurement_time(std::time::Duration::from_secs(3));
    targets = bench_argmax_softmax
);

criterion_main!(softmax_benches);
