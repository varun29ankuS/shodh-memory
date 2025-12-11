//! Streaming Memory Ingestion Benchmarks
//!
//! Benchmarks for the streaming memory pipeline:
//! 1. Session creation and management
//! 2. Message buffering and deduplication
//! 3. Content hashing performance
//! 4. NER extraction on streaming content
//! 5. Importance calculation
//! 6. Full extraction pipeline throughput
//!
//! Run with: cargo bench --bench streaming_benchmarks

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{MemoryConfig, MemorySystem};
use shodh_memory::streaming::{
    ExtractionConfig, StreamHandshake, StreamMessage, StreamMode, StreamingMemoryExtractor,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;
use tokio::runtime::Runtime;

// ==============================================================================
// Test Data: Realistic Streaming Content
// ==============================================================================

/// Conversation messages (LLM agent dialogue)
const CONVERSATION_MESSAGES: &[&str] = &[
    "User asked about weather in San Francisco today",
    "The forecast shows sunny with temperatures around 65°F in downtown San Francisco",
    "User mentioned they have a meeting with John Smith at Google headquarters",
    "Recommended taking BART from Powell Street station to Mountain View",
    "User confirmed they will leave at 2pm for the meeting",
    "Created calendar reminder for Google meeting with John Smith",
    "User asked to remember their preference for window seats on flights",
    "Noted: User prefers window seats on all flight bookings",
    "User mentioned allergies to shellfish - important for restaurant recommendations",
    "Flagged food allergy: shellfish - will exclude from dining suggestions",
];

/// Sensor readings (robotics/IoT)
const SENSOR_READINGS: &[(&str, f64, &str)] = &[
    ("temperature", 23.5, "°C"),
    ("humidity", 65.2, "%"),
    ("pressure", 1013.25, "hPa"),
    ("battery", 78.0, "%"),
    ("cpu_load", 45.3, "%"),
    ("memory_usage", 2048.0, "MB"),
    ("network_latency", 12.5, "ms"),
    ("gps_accuracy", 3.2, "m"),
    ("altitude", 125.8, "m"),
    ("speed", 0.5, "m/s"),
];

/// Event messages (system logs)
const EVENT_MESSAGES: &[(&str, &str, &str)] = &[
    ("info", "startup", "System initialized successfully"),
    ("debug", "connection", "Connected to database server"),
    ("warning", "memory", "Memory usage exceeded 80% threshold"),
    ("error", "network", "Connection timeout to API endpoint"),
    ("info", "task", "Background job completed: data sync"),
    ("decision", "routing", "Selected path A based on traffic analysis"),
    ("discovery", "pattern", "Detected recurring user behavior pattern"),
    ("learning", "model", "Updated preference model with new data"),
    ("error", "validation", "Invalid input format received from user"),
    ("info", "cleanup", "Garbage collection freed 256MB"),
];

/// Long-form content for extraction testing
const LONG_CONTENT: &str = "During the quarterly planning session at Microsoft headquarters in Redmond, \
    Satya Nadella discussed the strategic partnership with OpenAI. The collaboration has led to \
    significant advancements in Azure's AI capabilities. Engineers from both companies worked together \
    in Seattle and San Francisco offices to integrate GPT-4 into Microsoft products. The team lead, \
    Sarah Chen, presented results showing a 40% improvement in code completion accuracy. Next steps \
    include expanding to offices in London, Bangalore, and Singapore.";

// ==============================================================================
// Helper Functions
// ==============================================================================

fn setup_memory_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 200,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.3,
    };

    let memory_system = MemorySystem::new(config).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

fn setup_ner() -> Arc<NeuralNer> {
    let config = NerConfig::default();
    Arc::new(NeuralNer::new(config).expect("Failed to create NER"))
}

fn setup_fallback_ner() -> Arc<NeuralNer> {
    let config = NerConfig::default();
    Arc::new(NeuralNer::new_fallback(config))
}

fn create_handshake(mode: StreamMode) -> StreamHandshake {
    StreamHandshake {
        user_id: "benchmark_user".to_string(),
        mode,
        extraction_config: ExtractionConfig {
            min_importance: 0.3,
            auto_dedupe: true,
            dedupe_threshold: 0.85,
            checkpoint_interval_ms: 0, // Disable time-based for benchmarks
            max_buffer_size: 50,
            extract_entities: true,
            create_relationships: true,
            merge_consecutive: true,
            trigger_events: vec![
                "error".to_string(),
                "decision".to_string(),
                "discovery".to_string(),
            ],
        },
        session_id: None,
        metadata: HashMap::new(),
    }
}

fn create_content_message(content: &str) -> StreamMessage {
    StreamMessage::Content {
        content: content.to_string(),
        source: Some("assistant".to_string()),
        timestamp: None,
        importance: None,
        tags: vec![],
        metadata: HashMap::new(),
    }
}

fn create_event_message(severity: &str, event: &str, description: &str) -> StreamMessage {
    StreamMessage::Event {
        event: event.to_string(),
        description: description.to_string(),
        timestamp: None,
        severity: Some(severity.to_string()),
        data: HashMap::new(),
    }
}

fn create_sensor_message(sensor_id: &str, value: f64, unit: &str) -> StreamMessage {
    let mut values = HashMap::new();
    values.insert("reading".to_string(), value);

    let mut units = HashMap::new();
    units.insert("reading".to_string(), unit.to_string());

    StreamMessage::Sensor {
        sensor_id: sensor_id.to_string(),
        values,
        timestamp: None,
        units,
    }
}

// ==============================================================================
// BENCHMARK 1: Session Creation and Management
// ==============================================================================

fn bench_session_creation(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STREAMING BENCHMARK 1: Session Creation & Management        ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let rt = Runtime::new().unwrap();
    let ner = setup_ner();

    let mut group = c.benchmark_group("session_management");

    // Session creation
    group.bench_function("create_session", |b| {
        b.iter_batched(
            || {
                let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
                let handshake = create_handshake(StreamMode::Conversation);
                (extractor, handshake)
            },
            |(extractor, handshake)| rt.block_on(extractor.create_session(handshake)),
            BatchSize::SmallInput,
        );
    });

    // Different stream modes
    for mode in [StreamMode::Conversation, StreamMode::Sensor, StreamMode::Event] {
        let mode_name = match mode {
            StreamMode::Conversation => "conversation",
            StreamMode::Sensor => "sensor",
            StreamMode::Event => "event",
        };

        group.bench_function(BenchmarkId::new("create_session_mode", mode_name), |b| {
            b.iter_batched(
                || {
                    let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
                    let handshake = create_handshake(mode);
                    (extractor, handshake)
                },
                |(extractor, handshake)| rt.block_on(extractor.create_session(handshake)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ==============================================================================
// BENCHMARK 2: Content Hashing (Deduplication)
// ==============================================================================

fn bench_content_hashing(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STREAMING BENCHMARK 2: Content Hashing (Deduplication)      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut group = c.benchmark_group("content_hashing");

    // Short content
    group.bench_function("hash_short", |b| {
        let content = CONVERSATION_MESSAGES[0];
        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            content.to_lowercase().trim().hash(&mut hasher);
            hasher.finish()
        });
    });

    // Medium content
    group.bench_function("hash_medium", |b| {
        let content = CONVERSATION_MESSAGES.join(" ");
        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            content.to_lowercase().trim().hash(&mut hasher);
            hasher.finish()
        });
    });

    // Long content
    group.bench_function("hash_long", |b| {
        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            LONG_CONTENT.to_lowercase().trim().hash(&mut hasher);
            hasher.finish()
        });
    });

    // Dedup check with HashSet
    group.bench_function("dedup_check_100_items", |b| {
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for i in 0..100 {
            use std::hash::{Hash, Hasher};
            let content = format!("Test content {}", i);
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            content.hash(&mut hasher);
            seen.insert(hasher.finish());
        }

        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "New content to check".hash(&mut hasher);
            seen.contains(&hasher.finish())
        });
    });

    group.bench_function("dedup_check_1000_items", |b| {
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for i in 0..1000 {
            use std::hash::{Hash, Hasher};
            let content = format!("Test content {}", i);
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            content.hash(&mut hasher);
            seen.insert(hasher.finish());
        }

        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "New content to check".hash(&mut hasher);
            seen.contains(&hasher.finish())
        });
    });

    group.finish();
}

// ==============================================================================
// BENCHMARK 3: Importance Calculation
// ==============================================================================

fn bench_importance_calculation(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STREAMING BENCHMARK 3: Importance Calculation               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut group = c.benchmark_group("importance_calculation");

    // Test the importance heuristics
    let test_contents = vec![
        ("short_neutral", "Weather is nice today"),
        ("short_important", "CRITICAL: System failure detected"),
        ("with_entities", "Satya Nadella announced Microsoft partnership with OpenAI"),
        ("with_numbers", "Revenue increased by 45% to reach $50 billion"),
        ("error_content", "Error: Connection failed to database server unexpectedly"),
        ("decision_content", "Decision made to proceed with plan A for the deployment"),
    ];

    for (name, content) in test_contents {
        group.bench_function(BenchmarkId::new("calculate", name), |b| {
            b.iter(|| {
                let mut importance: f32 = 0.5;

                // Length penalty for very short content
                if content.len() < 20 {
                    importance -= 0.2;
                } else if content.len() > 100 {
                    importance += 0.1;
                }

                // Keyword boosts
                let lower = content.to_lowercase();
                if lower.contains("error") || lower.contains("critical") {
                    importance += 0.3;
                }
                if lower.contains("decision") || lower.contains("decided") {
                    importance += 0.2;
                }
                if lower.contains("important") || lower.contains("remember") {
                    importance += 0.2;
                }

                // Entity presence boost
                let has_capitalized = content
                    .split_whitespace()
                    .any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));
                if has_capitalized {
                    importance += 0.1;
                }

                // Number presence
                if content.chars().any(|c| c.is_ascii_digit()) {
                    importance += 0.1;
                }

                importance.clamp(0.0, 1.0)
            });
        });
    }

    group.finish();
}

// ==============================================================================
// BENCHMARK 4: Message Processing Throughput
// ==============================================================================

fn bench_message_processing(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STREAMING BENCHMARK 4: Message Processing Throughput        ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let rt = Runtime::new().unwrap();
    let ner = setup_ner();
    let (memory_system, _temp_dir) = setup_memory_system();
    let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));

    let mut group = c.benchmark_group("message_processing");
    group.throughput(Throughput::Elements(1));

    // Content message processing
    group.bench_function("process_content_message", |b| {
        let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
        let session_id = rt.block_on(extractor.create_session(create_handshake(StreamMode::Conversation)));

        b.iter(|| {
            let msg = create_content_message(CONVERSATION_MESSAGES[0]);
            rt.block_on(extractor.process_message(&session_id, msg, Arc::clone(&memory_arc)))
        });
    });

    // Event message processing
    group.bench_function("process_event_message", |b| {
        let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
        let session_id = rt.block_on(extractor.create_session(create_handshake(StreamMode::Event)));

        b.iter(|| {
            let (severity, event, desc) = EVENT_MESSAGES[0];
            let msg = create_event_message(severity, event, desc);
            rt.block_on(extractor.process_message(&session_id, msg, Arc::clone(&memory_arc)))
        });
    });

    // Sensor message processing
    group.bench_function("process_sensor_message", |b| {
        let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
        let session_id = rt.block_on(extractor.create_session(create_handshake(StreamMode::Sensor)));

        b.iter(|| {
            let (sensor_id, value, unit) = SENSOR_READINGS[0];
            let msg = create_sensor_message(sensor_id, value, unit);
            rt.block_on(extractor.process_message(&session_id, msg, Arc::clone(&memory_arc)))
        });
    });

    // Ping message (baseline)
    group.bench_function("process_ping", |b| {
        let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
        let session_id = rt.block_on(extractor.create_session(create_handshake(StreamMode::Conversation)));

        b.iter(|| {
            rt.block_on(extractor.process_message(
                &session_id,
                StreamMessage::Ping,
                Arc::clone(&memory_arc),
            ))
        });
    });

    group.finish();
}

// ==============================================================================
// BENCHMARK 5: Batch Message Throughput
// ==============================================================================

fn bench_batch_throughput(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STREAMING BENCHMARK 5: Batch Message Throughput             ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let rt = Runtime::new().unwrap();
    let ner = setup_ner();

    let mut group = c.benchmark_group("batch_throughput");

    // Batch sizes to test
    for batch_size in [10, 25, 50, 100] {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("conversation_batch", batch_size),
            &batch_size,
            |b, &size| {
                let (memory_system, _temp_dir) = setup_memory_system();
                let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));
                let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
                let session_id = rt.block_on(extractor.create_session(create_handshake(StreamMode::Conversation)));

                // Pre-generate messages
                let messages: Vec<StreamMessage> = (0..size)
                    .map(|i| {
                        let content = format!(
                            "{} - iteration {}",
                            CONVERSATION_MESSAGES[i % CONVERSATION_MESSAGES.len()],
                            i
                        );
                        create_content_message(&content)
                    })
                    .collect();

                b.iter(|| {
                    for msg in messages.clone() {
                        rt.block_on(extractor.process_message(
                            &session_id,
                            msg,
                            Arc::clone(&memory_arc),
                        ));
                    }
                    // Flush at end
                    rt.block_on(extractor.process_message(
                        &session_id,
                        StreamMessage::Flush,
                        Arc::clone(&memory_arc),
                    ))
                });
            },
        );
    }

    group.finish();
}

// ==============================================================================
// BENCHMARK 6: NER on Streaming Content
// ==============================================================================

fn bench_streaming_ner(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STREAMING BENCHMARK 6: NER on Streaming Content             ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let ner = setup_ner();
    let fallback_ner = setup_fallback_ner();

    // Warm up neural NER
    let _ = ner.extract("Warmup text");

    let mut group = c.benchmark_group("streaming_ner");

    // Conversation content NER
    group.bench_function("neural_conversation", |b| {
        b.iter(|| {
            for msg in CONVERSATION_MESSAGES {
                let _ = ner.extract(msg);
            }
        });
    });

    group.bench_function("fallback_conversation", |b| {
        b.iter(|| {
            for msg in CONVERSATION_MESSAGES {
                let _ = fallback_ner.extract(msg);
            }
        });
    });

    // Long content NER
    group.bench_function("neural_long_content", |b| {
        b.iter(|| ner.extract(LONG_CONTENT));
    });

    group.bench_function("fallback_long_content", |b| {
        b.iter(|| fallback_ner.extract(LONG_CONTENT));
    });

    // Combined buffer NER (multiple messages joined)
    group.bench_function("neural_buffer_10_messages", |b| {
        let combined = CONVERSATION_MESSAGES.join("\n");
        b.iter(|| ner.extract(&combined));
    });

    group.finish();

    // Print entity extraction summary
    eprintln!("\n📊 STREAMING NER SUMMARY:");
    if let Ok(entities) = ner.extract(LONG_CONTENT) {
        eprintln!("   Long content ({} chars): {} entities", LONG_CONTENT.len(), entities.len());
        for e in &entities {
            eprintln!("     - {} ({:?}, {:.2})", e.text, e.entity_type, e.confidence);
        }
    }
}

// ==============================================================================
// BENCHMARK 7: Full Extraction Pipeline
// ==============================================================================

fn bench_full_extraction_pipeline(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STREAMING BENCHMARK 7: Full Extraction Pipeline             ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let rt = Runtime::new().unwrap();
    let ner = setup_ner();

    let mut group = c.benchmark_group("full_extraction_pipeline");

    // Complete pipeline: create session -> buffer messages -> flush -> extract
    group.bench_function("complete_pipeline_10_messages", |b| {
        b.iter_batched(
            || {
                let (memory_system, temp_dir) = setup_memory_system();
                let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));
                let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
                (extractor, memory_arc, temp_dir)
            },
            |(extractor, memory_arc, _temp_dir)| {
                rt.block_on(async {
                    // Create session
                    let session_id = extractor.create_session(create_handshake(StreamMode::Conversation)).await;

                    // Buffer messages
                    for msg in CONVERSATION_MESSAGES {
                        let _ = extractor
                            .process_message(
                                &session_id,
                                create_content_message(msg),
                                Arc::clone(&memory_arc),
                            )
                            .await;
                    }

                    // Flush and extract
                    extractor
                        .process_message(&session_id, StreamMessage::Flush, memory_arc)
                        .await
                })
            },
            BatchSize::SmallInput,
        );
    });

    // Event-triggered extraction
    group.bench_function("event_triggered_extraction", |b| {
        b.iter_batched(
            || {
                let (memory_system, temp_dir) = setup_memory_system();
                let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));
                let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));
                (extractor, memory_arc, temp_dir)
            },
            |(extractor, memory_arc, _temp_dir)| {
                rt.block_on(async {
                    let session_id = extractor.create_session(create_handshake(StreamMode::Event)).await;

                    // Buffer some events
                    for (severity, event, desc) in &EVENT_MESSAGES[..5] {
                        let _ = extractor
                            .process_message(
                                &session_id,
                                create_event_message(severity, event, desc),
                                Arc::clone(&memory_arc),
                            )
                            .await;
                    }

                    // Send trigger event (error)
                    extractor
                        .process_message(
                            &session_id,
                            create_event_message("error", "error", "Critical failure detected"),
                            memory_arc,
                        )
                        .await
                })
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ==============================================================================
// BENCHMARK 8: Streaming Performance Summary
// ==============================================================================

fn bench_streaming_summary(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                           STREAMING PERFORMANCE SUMMARY                                       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    let rt = Runtime::new().unwrap();
    let ner = setup_ner();

    // Measure each component
    let iterations = 10;

    let mut session_times = Vec::new();
    let mut buffer_times = Vec::new();
    let mut flush_times = Vec::new();
    let mut ner_times = Vec::new();

    for _ in 0..iterations {
        let (memory_system, _temp_dir) = setup_memory_system();
        let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));
        let extractor = StreamingMemoryExtractor::new(Arc::clone(&ner));

        // Session creation
        let start = Instant::now();
        let session_id = rt.block_on(extractor.create_session(create_handshake(StreamMode::Conversation)));
        session_times.push(start.elapsed());

        // Message buffering
        let start = Instant::now();
        for msg in CONVERSATION_MESSAGES {
            rt.block_on(extractor.process_message(
                &session_id,
                create_content_message(msg),
                Arc::clone(&memory_arc),
            ));
        }
        buffer_times.push(start.elapsed());

        // Flush/extraction
        let start = Instant::now();
        rt.block_on(extractor.process_message(
            &session_id,
            StreamMessage::Flush,
            Arc::clone(&memory_arc),
        ));
        flush_times.push(start.elapsed());

        // NER on combined content
        let start = Instant::now();
        let _ = ner.extract(LONG_CONTENT);
        ner_times.push(start.elapsed());
    }

    // Calculate averages
    let avg_session = session_times.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_buffer = buffer_times.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_flush = flush_times.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_ner = ner_times.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;

    let total = avg_session + avg_buffer + avg_flush;
    let per_message = avg_buffer / CONVERSATION_MESSAGES.len() as f64;

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           STREAMING PIPELINE BREAKDOWN                                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  OPERATION                   │  AVG TIME     │  NOTES                                         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Session Creation            │  {:>8.2}ms   │  One-time per connection                       ║", avg_session);
    println!("║  Buffer 10 Messages          │  {:>8.2}ms   │  Includes dedup check                          ║", avg_buffer);
    println!("║  Per Message (buffering)     │  {:>8.2}ms   │  Hash + dedup + buffer push                    ║", per_message);
    println!("║  Flush + Extract             │  {:>8.2}ms   │  NER + Memory creation                         ║", avg_flush);
    println!("║  NER (long content)          │  {:>8.2}ms   │  Neural NER on paragraph                       ║", avg_ner);
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  TOTAL (10 messages)         │  {:>8.2}ms   │  Session + Buffer + Flush                      ║", total);
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Throughput summary
    let msgs_per_sec = (CONVERSATION_MESSAGES.len() as f64 / (total / 1000.0)).round() as u64;
    println!("📊 THROUGHPUT: ~{} messages/second (including extraction)", msgs_per_sec);
    println!();

    // Edge device estimates
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           EDGE DEVICE ESTIMATES                                               ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  DEVICE               │  MSG/SEC (EST)  │  BUFFER LATENCY  │  USE CASE                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Raspberry Pi 4       │     50-100      │     10-20ms      │  Home automation, IoT            ║");
    println!("║  Jetson Nano          │    100-200      │      5-10ms      │  Robotics, drones                ║");
    println!("║  Intel NUC            │    200-400      │      2-5ms       │  Edge server, kiosk              ║");
    println!("║  Desktop/Cloud        │    500-1000     │      1-2ms       │  Development, production         ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Dummy benchmark to satisfy criterion
    c.bench_function("zzz_streaming_summary", |b| {
        b.iter(|| std::hint::black_box(1 + 1))
    });
}

// ==============================================================================
// CRITERION CONFIGURATION
// ==============================================================================

criterion_group!(
    name = streaming_benches;
    config = Criterion::default()
        .sample_size(30)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_session_creation,
        bench_content_hashing,
        bench_importance_calculation,
        bench_message_processing,
        bench_batch_throughput,
        bench_streaming_ner,
        bench_full_extraction_pipeline,
        bench_streaming_summary
);

criterion_main!(streaming_benches);
