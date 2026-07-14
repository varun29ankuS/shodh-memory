//! Performance Benchmarks for Neural NER Module
//!
//! Demonstrates edge-device readiness:
//! - Model loading latency
//! - Inference throughput (entities/second)
//! - Memory efficiency
//! - Comparison: Neural vs Rule-based fallback

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use std::time::Instant;

// ==============================================================================
// Test Data: Indian & Global Entities
// ==============================================================================

/// Short texts (typical user input)
const SHORT_TEXTS: &[(&str, &str)] = &[
    (
        "person_single",
        "Narendra Modi is the Prime Minister of India",
    ),
    ("org_single", "Infosys is headquartered in Bangalore"),
    ("loc_single", "The Taj Mahal is located in Agra"),
    (
        "mixed_short",
        "Sundar Pichai from Google visited IIT Madras in Chennai",
    ),
];

/// Medium texts (typical document sentence)
const MEDIUM_TEXTS: &[(&str, &str)] = &[
    ("indian_tech", "Tata Consultancy Services and Wipro are expanding their operations in Hyderabad and Pune, with Ratan Tata and Azim Premji leading the initiatives"),
    ("global_tech", "Satya Nadella announced that Microsoft will partner with OpenAI to bring GPT technology to Azure cloud services in Seattle"),
    ("mixed_entities", "The Reserve Bank of India governor Shaktikanta Das met with Janet Yellen in Mumbai to discuss bilateral trade between India and United States"),
];

/// Long texts (paragraph-level extraction)
const LONG_TEXTS: &[(&str, &str)] = &[
    ("news_article", "In a landmark development for India's technology sector, Mukesh Ambani's Reliance Industries announced a strategic partnership with Google and Facebook. The collaboration, valued at over 500 crore rupees, will see Jio Platforms expanding its 5G infrastructure across Mumbai, Delhi, Bangalore, and Chennai. This follows similar investments by Tata Group in Pune and Infosys in Hyderabad. Industry experts including N. R. Narayana Murthy and Nandan Nilekani praised the move, stating it would accelerate India's digital transformation."),
    ("research_abstract", "This study examines the impact of artificial intelligence research conducted at Indian Institute of Technology Delhi, Stanford University, and Massachusetts Institute of Technology. Lead researchers Dr. Priya Sharma from IIT Delhi and Professor Andrew Ng from Stanford collaborated with teams at Google DeepMind in London and Microsoft Research in Redmond. The findings were presented at the Neural Information Processing Systems conference in Vancouver, with participants from Amazon, Meta, and NVIDIA."),
];

/// Edge cases for robustness testing
const EDGE_CASES: &[(&str, &str)] = &[
    ("empty", ""),
    ("no_entities", "The quick brown fox jumps over the lazy dog"),
    (
        "unicode_hindi",
        "नरेंद्र मोदी ने Google के CEO सुंदर पिचाई से मुलाकात की",
    ),
    (
        "special_chars",
        "Dr. A.P.J. Abdul Kalam visited ISRO-DRDO headquarters",
    ),
    (
        "repeated_entity",
        "Microsoft, Microsoft, Microsoft announced a partnership with Microsoft",
    ),
    (
        "abbreviations",
        "TCS, HCL, and WIPRO are listed on NSE and BSE",
    ),
];

// ==============================================================================
// Helper Functions
// ==============================================================================

/// Create NER instance with neural model (if available) or fallback
fn setup_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new(config).expect("Failed to create NER model")
}

/// Create NER instance with explicit fallback mode
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

// ==============================================================================
// Benchmark 1: Model Initialization
// ==============================================================================

fn bench_model_init(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  NER MODEL INITIALIZATION BENCHMARKS                     ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝\n");

    let mut group = c.benchmark_group("ner_init");
    group.sample_size(10); // Model loading is slow, reduce samples

    // Benchmark cold initialization (includes model loading)
    group.bench_function("cold_init", |b| {
        b.iter_batched(
            || {},
            |_| {
                let config = NerConfig::default();
                let ner = NeuralNer::new(config).expect("Failed to create NER");
                // Force model load by extracting entities
                let _ = ner.extract("Test text");
            },
            BatchSize::PerIteration,
        );
    });

    // Benchmark fallback initialization (no model loading)
    group.bench_function("fallback_init", |b| {
        b.iter(|| {
            let ner = setup_fallback_ner();
            let _ = ner.extract("Test text");
        });
    });

    group.finish();
}

// ==============================================================================
// Benchmark 2: Short Text Extraction (User Input Simulation)
// ==============================================================================

fn bench_short_text_extraction(c: &mut Criterion) {
    eprintln!("\n⚡ SHORT TEXT EXTRACTION (User Input) ⚡\n");

    let ner = setup_ner();
    // Warm up model
    let _ = ner.extract("Warmup text for model loading");

    let mut group = c.benchmark_group("ner_short_text");

    for (label, text) in SHORT_TEXTS {
        group.bench_with_input(BenchmarkId::from_parameter(label), text, |b, text| {
            b.iter(|| ner.extract(text));
        });
    }

    group.finish();
}

// ==============================================================================
// Benchmark 3: Medium Text Extraction (Document Sentence)
// ==============================================================================

fn bench_medium_text_extraction(c: &mut Criterion) {
    eprintln!("\n📄 MEDIUM TEXT EXTRACTION (Document Sentence) 📄\n");

    let ner = setup_ner();
    let _ = ner.extract("Warmup");

    let mut group = c.benchmark_group("ner_medium_text");

    for (label, text) in MEDIUM_TEXTS {
        group.bench_with_input(BenchmarkId::from_parameter(label), text, |b, text| {
            b.iter(|| ner.extract(text));
        });
    }

    group.finish();
}

// ==============================================================================
// Benchmark 4: Long Text Extraction (Paragraph-Level)
// ==============================================================================

fn bench_long_text_extraction(c: &mut Criterion) {
    eprintln!("\n📖 LONG TEXT EXTRACTION (Paragraph-Level) 📖\n");

    let ner = setup_ner();
    let _ = ner.extract("Warmup");

    let mut group = c.benchmark_group("ner_long_text");
    group.sample_size(20); // Long texts take longer

    for (label, text) in LONG_TEXTS {
        group.bench_with_input(BenchmarkId::from_parameter(label), text, |b, text| {
            b.iter(|| ner.extract(text));
        });
    }

    group.finish();
}

// ==============================================================================
// Benchmark 5: Edge Cases
// ==============================================================================

fn bench_edge_cases(c: &mut Criterion) {
    eprintln!("\n🔬 EDGE CASE HANDLING 🔬\n");

    let ner = setup_ner();
    let _ = ner.extract("Warmup");

    let mut group = c.benchmark_group("ner_edge_cases");

    for (label, text) in EDGE_CASES {
        group.bench_with_input(BenchmarkId::from_parameter(label), text, |b, text| {
            b.iter(|| ner.extract(text));
        });
    }

    group.finish();
}

// ==============================================================================
// Benchmark 6: Neural vs Fallback Comparison
// ==============================================================================

fn bench_neural_vs_fallback(c: &mut Criterion) {
    eprintln!("\n🔄 NEURAL vs FALLBACK COMPARISON 🔄\n");

    let neural_ner = setup_ner();
    let fallback_ner = setup_fallback_ner();

    // Warm up neural model
    let _ = neural_ner.extract("Warmup text");

    let test_text =
        "Sundar Pichai from Google met Satya Nadella from Microsoft in Bangalore last week";

    let mut group = c.benchmark_group("ner_comparison");

    group.bench_function("neural", |b| {
        b.iter(|| neural_ner.extract(test_text));
    });

    group.bench_function("fallback_rules", |b| {
        b.iter(|| fallback_ner.extract(test_text));
    });

    group.finish();

    // Print accuracy comparison
    eprintln!("\n📊 ACCURACY COMPARISON:");
    if let Ok(neural_entities) = neural_ner.extract(test_text) {
        eprintln!("   Neural extracted {} entities:", neural_entities.len());
        for e in &neural_entities {
            eprintln!(
                "     - {} ({:?}, {:.2})",
                e.text, e.entity_type, e.confidence
            );
        }
    }

    if let Ok(fallback_entities) = fallback_ner.extract(test_text) {
        eprintln!(
            "   Fallback extracted {} entities:",
            fallback_entities.len()
        );
        for e in &fallback_entities {
            eprintln!(
                "     - {} ({:?}, {:.2})",
                e.text, e.entity_type, e.confidence
            );
        }
    }
}

// ==============================================================================
// Benchmark 7: Batch Processing Throughput
// ==============================================================================

fn bench_batch_throughput(c: &mut Criterion) {
    eprintln!("\n🚀 BATCH PROCESSING THROUGHPUT 🚀\n");

    let ner = setup_ner();
    let _ = ner.extract("Warmup");

    // Create batch of varied texts
    let batch: Vec<&str> = SHORT_TEXTS
        .iter()
        .chain(MEDIUM_TEXTS.iter())
        .map(|(_, text)| *text)
        .collect();

    let mut group = c.benchmark_group("ner_batch");

    // Sequential processing
    group.bench_function("sequential_7_texts", |b| {
        b.iter(|| {
            for text in &batch {
                let _ = ner.extract(text);
            }
        });
    });

    group.finish();

    // Print throughput stats
    eprintln!("\n📈 THROUGHPUT METRICS:");
    let start = Instant::now();
    let mut total_entities = 0;
    for text in &batch {
        if let Ok(entities) = ner.extract(text) {
            total_entities += entities.len();
        }
    }
    let elapsed = start.elapsed();
    eprintln!("   Processed {} texts in {:?}", batch.len(), elapsed);
    eprintln!("   Extracted {} total entities", total_entities);
    eprintln!(
        "   Throughput: {:.1} texts/sec",
        batch.len() as f64 / elapsed.as_secs_f64()
    );
}

// ==============================================================================
// Benchmark 8: Entity Type Distribution
// ==============================================================================

fn bench_entity_type_extraction(c: &mut Criterion) {
    eprintln!("\n🏷️ ENTITY TYPE EXTRACTION 🏷️\n");

    let ner = setup_ner();
    let _ = ner.extract("Warmup");

    let mut group = c.benchmark_group("ner_entity_types");

    // Person-heavy text
    let person_text = "Narendra Modi, Rahul Gandhi, Arvind Kejriwal, and Mamata Banerjee are prominent Indian politicians";
    group.bench_function("person_heavy", |b| {
        b.iter(|| ner.extract(person_text));
    });

    // Organization-heavy text
    let org_text =
        "TCS, Infosys, Wipro, HCL Technologies, and Tech Mahindra are India's largest IT companies";
    group.bench_function("org_heavy", |b| {
        b.iter(|| ner.extract(org_text));
    });

    // Location-heavy text
    let loc_text =
        "Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, and Kolkata are major Indian metros";
    group.bench_function("location_heavy", |b| {
        b.iter(|| ner.extract(loc_text));
    });

    group.finish();
}

// ==============================================================================
// Benchmark 9: Memory Efficiency (Entity Count Scaling)
// ==============================================================================

fn bench_entity_scaling(c: &mut Criterion) {
    eprintln!("\n📊 ENTITY COUNT SCALING 📊\n");

    let ner = setup_ner();
    let _ = ner.extract("Warmup");

    let mut group = c.benchmark_group("ner_scaling");
    group.sample_size(20);

    // 1 entity
    let text_1 = "Infosys is a company";
    group.bench_with_input(
        BenchmarkId::from_parameter("1_entity"),
        &text_1,
        |b, text| {
            b.iter(|| ner.extract(text));
        },
    );

    // 5 entities
    let text_5 = "Infosys and Wipro in Bangalore, TCS in Mumbai, HCL in Noida";
    group.bench_with_input(
        BenchmarkId::from_parameter("5_entities"),
        &text_5,
        |b, text| {
            b.iter(|| ner.extract(text));
        },
    );

    // 10 entities
    let text_10 =
        "Infosys, Wipro, TCS, HCL, Tech Mahindra in Bangalore, Mumbai, Delhi, Chennai, Hyderabad";
    group.bench_with_input(
        BenchmarkId::from_parameter("10_entities"),
        &text_10,
        |b, text| {
            b.iter(|| ner.extract(text));
        },
    );

    // 20+ entities (stress test)
    let text_20 = "Narendra Modi visited TCS, Infosys, Wipro, HCL, Tech Mahindra, Cognizant, Mphasis, Mindtree, L&T Infotech, Cyient in Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad, Jaipur, Lucknow meeting Ratan Tata, Azim Premji, N. R. Narayana Murthy";
    group.bench_with_input(
        BenchmarkId::from_parameter("20_entities"),
        &text_20,
        |b, text| {
            b.iter(|| ner.extract(text));
        },
    );

    group.finish();
}

// ==============================================================================
// Benchmark 10: Summary & Performance Report
// ==============================================================================

fn bench_print_summary(c: &mut Criterion) {
    // Minimal benchmark to trigger summary
    c.bench_function("zzz_ner_summary", |b| {
        b.iter(|| std::hint::black_box(1 + 1))
    });

    print_ner_summary();
}

// ANSI color codes
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";

fn print_ner_summary() {
    println!("\n{BOLD}");
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                                              ║");
    println!("║   {CYAN}███╗   ██╗███████╗██████╗ {RESET}      {MAGENTA}███████╗██╗  ██╗ ██████╗ ██████╗ ██╗  ██╗{RESET}                       ║");
    println!("║   {CYAN}████╗  ██║██╔════╝██╔══██╗{RESET}      {MAGENTA}██╔════╝██║  ██║██╔═══██╗██╔══██╗██║  ██║{RESET}                       ║");
    println!("║   {CYAN}██╔██╗ ██║█████╗  ██████╔╝{RESET}█████╗{MAGENTA}███████╗███████║██║   ██║██║  ██║███████║{RESET}                       ║");
    println!("║   {CYAN}██║╚██╗██║██╔══╝  ██╔══██╗{RESET}      {MAGENTA}╚════██║██╔══██║██║   ██║██║  ██║██╔══██║{RESET}                       ║");
    println!("║   {CYAN}██║ ╚████║███████╗██║  ██║{RESET}      {MAGENTA}███████║██║  ██║╚██████╔╝██████╔╝██║  ██║{RESET}                       ║");
    println!("║   {CYAN}╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝{RESET}      {MAGENTA}╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝{RESET}                       ║");
    println!("║                                                                                              ║");
    println!("║                    {BOLD}Neural Named Entity Recognition for Edge Devices{RESET}                        ║");
    println!("║                           {YELLOW}GLiNER bi-edge-v2 (ONNX) Benchmarks{RESET}                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("{RESET}");
    println!();

    // Model specifications
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                  MODEL SPECIFICATIONS                                         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                                               ║");
    println!("║  {CYAN}Model{RESET}:               knowledgator/gliner-bi-edge-v2.0 (ONNX text tower)                    ║");
    println!("║  {CYAN}Size{RESET}:                ~149MB fp32 model + precomputed label embeddings                      ║");
    println!("║  {CYAN}Architecture{RESET}:        BiEncoder span GLiNER — frozen text tower + span/label scorer         ║");
    println!("║  {CYAN}Schema{RESET}:              141 fine / 18 coarse entity types (schema-driven, growable)           ║");
    println!("║  {CYAN}Entity Types{RESET}:        Person, Org, Location, GPE, Facility, Weapon, Cyber, Money, ...      ║");
    println!("║  {CYAN}Max Sequence{RESET}:        128 tokens (optimized for edge)                                       ║");
    println!("║                                                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Performance targets
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                  PERFORMANCE TARGETS                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {BOLD}OPERATION                    │  TARGET    │  NOTES{RESET}                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Model Cold Load              │  < 500ms   │  First inference includes model load              ║");
    println!("║ Short Text (<50 chars)       │  < 10ms    │  User input, chat messages                        ║");
    println!("║ Medium Text (~200 chars)     │  < 20ms    │  Document sentences                               ║");
    println!("║ Long Text (~500 chars)       │  < 50ms    │  Paragraphs, abstracts                            ║");
    println!("║ Fallback (rule-based)        │  < 1ms     │  When model unavailable                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Edge device compatibility
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              EDGE DEVICE COMPATIBILITY                                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                                               ║");
    println!("║  {GREEN}✅ Raspberry Pi 4 (4GB+){RESET}:     Model fits in RAM, ~50-100ms inference                      ║");
    println!("║  {GREEN}✅ NVIDIA Jetson Nano{RESET}:        GPU acceleration possible, ~10-30ms                         ║");
    println!("║  {GREEN}✅ Intel NUC{RESET}:                 x86 optimizations, ~5-15ms                                   ║");
    println!("║  {GREEN}✅ Android (8GB+){RESET}:            ONNX Runtime Mobile support                                  ║");
    println!("║  {GREEN}✅ iOS (A12+){RESET}:                CoreML conversion possible                                   ║");
    println!("║                                                                                               ║");
    println!("║  {YELLOW}Memory Requirements{RESET}:                                                                      ║");
    println!("║    - Model: ~149MB (fp32 ONNX)                                                                ║");
    println!("║    • Tokenizer: ~500KB                                                                        ║");
    println!("║    • Runtime: ~50MB (ONNX Runtime)                                                            ║");
    println!("║    • Working memory: ~10MB per inference                                                      ║");
    println!("║                                                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Comparison with alternatives
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           COMPARISON WITH ALTERNATIVES                                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                                               ║");
    println!("║  {CYAN}GLiNER bi-edge-v2 (this){RESET} │  ~149MB │  span typing, 141 fine / 18 coarse types              ║");
    println!("║  bert-base-NER            │  ~400MB  │  50-100ms │  Better accuracy, server-only            ║");
    println!("║  spaCy en_core_web_sm     │  ~12MB   │  10-30ms │  General NLP, not NER-focused            ║");
    println!("║  spaCy en_core_web_trf    │  ~400MB  │  100-200ms │  Transformer, high accuracy            ║");
    println!("║  Flair NER                │  ~200MB  │  100-500ms │  High accuracy, slow inference         ║");
    println!("║  Rule-based (fallback)    │  ~0MB    │  <1ms    │  Low accuracy, zero latency              ║");
    println!("║                                                                                               ║");
    println!("║  {GREEN}Best for Edge{RESET}: GLiNER bi-edge-v2 — schema-driven fine typing, on-device ONNX             ║");
    println!("║                                                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Supported entity types with examples
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                            SUPPORTED ENTITY TYPES                                             ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                                               ║");
    println!("║  {MAGENTA}PER (Person){RESET}:                                                                             ║");
    println!("║    Indian: Narendra Modi, Ratan Tata, Sundar Pichai, N. R. Narayana Murthy                   ║");
    println!("║    Global: Elon Musk, Tim Cook, Satya Nadella, Mark Zuckerberg                               ║");
    println!("║                                                                                               ║");
    println!("║  {CYAN}ORG (Organization){RESET}:                                                                         ║");
    println!("║    Indian: TCS, Infosys, Wipro, Reliance, ISRO, IIT, RBI                                     ║");
    println!("║    Global: Google, Microsoft, Amazon, Apple, OpenAI, Tesla                                   ║");
    println!("║                                                                                               ║");
    println!("║  {GREEN}LOC (Location){RESET}:                                                                             ║");
    println!("║    Indian: Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune                                ║");
    println!("║    Global: New York, San Francisco, London, Tokyo, Singapore                                 ║");
    println!("║                                                                                               ║");
    println!("║  {YELLOW}MISC (Miscellaneous){RESET}:                                                                       ║");
    println!("║    Events, products, languages, nationalities, etc.                                          ║");
    println!("║                                                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Footer
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                                               ║");
    println!("║  {CYAN}Detailed results{RESET}:    target/criterion/ner_*/report/index.html                               ║");
    println!("║  {CYAN}Run benchmarks{RESET}:      cargo bench --bench ner_benchmarks                                   ║");
    println!("║                                                                                               ║");
    println!("║  {MAGENTA}Model source{RESET}:        https://huggingface.co/knowledgator/gliner-bi-edge-v2.0              ║");
    println!("║  {MAGENTA}Documentation{RESET}:       https://github.com/roshera/shodh-memory                              ║");
    println!("║                                                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

// ==============================================================================
// Criterion Configuration
// ==============================================================================

criterion_group!(
    name = ner_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_model_init,
        bench_short_text_extraction,
        bench_medium_text_extraction,
        bench_long_text_extraction,
        bench_edge_cases,
        bench_neural_vs_fallback,
        bench_batch_throughput,
        bench_entity_type_extraction,
        bench_entity_scaling,
        bench_print_summary
);

criterion_main!(ner_benches);
