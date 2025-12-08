//! Neural NER Integration Tests
//!
//! These tests download the TinyBERT-finetuned-NER model and run entity extraction.
//! Requires network access on first run to download the model (~14.5MB quantized).

use shodh_memory::embeddings::ner::{NerConfig, NerEntityType, NeuralNer};
use shodh_memory::embeddings::{
    are_ner_models_downloaded, download_ner_models, get_ner_models_dir,
};
use std::sync::Arc;

/// Download NER models if not already present
fn ensure_models_downloaded() {
    if !are_ner_models_downloaded() {
        println!("Downloading NER models (~17MB)...");
        let callback: Arc<dyn Fn(u64, u64) + Send + Sync> = Arc::new(|downloaded, total| {
            if total > 0 {
                let pct = (downloaded as f64 / total as f64) * 100.0;
                print!(
                    "\rDownloading: {:.1}% ({}/{} bytes)",
                    pct, downloaded, total
                );
            }
        });
        let result = download_ner_models(Some(callback));

        match result {
            Ok(path) => println!("\nModels downloaded to: {:?}", path),
            Err(e) => panic!("Failed to download NER models: {}", e),
        }
    } else {
        println!(
            "NER models already downloaded at: {:?}",
            get_ner_models_dir()
        );
    }
}

/// Create neural NER instance (downloads model if needed)
fn create_neural_ner() -> NeuralNer {
    ensure_models_downloaded();

    let models_dir = get_ner_models_dir();
    let config = NerConfig {
        model_path: models_dir.join("model.onnx"),
        tokenizer_path: models_dir.join("tokenizer.json"),
        max_length: 128,
        confidence_threshold: 0.5,
    };

    NeuralNer::new(config).expect("Failed to create neural NER")
}

// ==================== Neural NER Tests ====================

#[test]
fn test_neural_download_and_init() {
    let ner = create_neural_ner();
    assert!(
        !ner.is_fallback_mode(),
        "Should be using neural model, not fallback"
    );
    println!("Neural NER initialized successfully");
}

#[test]
fn test_neural_person_extraction() {
    let ner = create_neural_ner();

    let text = "Sundar Pichai is the CEO of Google";
    let entities = ner.extract(text).expect("Extraction failed");

    println!("Text: {}", text);
    println!("Entities found:");
    for e in &entities {
        println!(
            "  - {} ({:?}, confidence: {:.2})",
            e.text, e.entity_type, e.confidence
        );
    }

    // Neural model should find at least one entity
    assert!(!entities.is_empty(), "Should extract at least one entity");
}

#[test]
fn test_neural_org_extraction() {
    let ner = create_neural_ner();

    let text = "Microsoft and Google are tech companies";
    let entities = ner.extract(text).expect("Extraction failed");

    println!("Text: {}", text);
    println!("Entities found:");
    for e in &entities {
        println!(
            "  - {} ({:?}, confidence: {:.2})",
            e.text, e.entity_type, e.confidence
        );
    }

    let org_count = entities
        .iter()
        .filter(|e| e.entity_type == NerEntityType::Organization)
        .count();

    assert!(org_count >= 1, "Should find at least 1 organization");
}

#[test]
fn test_neural_location_extraction() {
    let ner = create_neural_ner();

    let text = "The office is located in Bangalore, India";
    let entities = ner.extract(text).expect("Extraction failed");

    println!("Text: {}", text);
    println!("Entities found:");
    for e in &entities {
        println!(
            "  - {} ({:?}, confidence: {:.2})",
            e.text, e.entity_type, e.confidence
        );
    }

    let loc_count = entities
        .iter()
        .filter(|e| e.entity_type == NerEntityType::Location)
        .count();

    assert!(loc_count >= 1, "Should find at least 1 location");
}

#[test]
fn test_neural_mixed_entities() {
    let ner = create_neural_ner();

    let text =
        "Satya Nadella announced that Microsoft will expand operations in Seattle and London";
    let entities = ner.extract(text).expect("Extraction failed");

    println!("Text: {}", text);
    println!("Entities found ({} total):", entities.len());
    for e in &entities {
        println!(
            "  - {} ({:?}, confidence: {:.2}, span: {}..{})",
            e.text, e.entity_type, e.confidence, e.start, e.end
        );
    }

    // Should find multiple entity types
    let has_person = entities
        .iter()
        .any(|e| e.entity_type == NerEntityType::Person);
    let has_org = entities
        .iter()
        .any(|e| e.entity_type == NerEntityType::Organization);
    let has_loc = entities
        .iter()
        .any(|e| e.entity_type == NerEntityType::Location);

    println!(
        "\nEntity types found: Person={}, Org={}, Location={}",
        has_person, has_org, has_loc
    );

    assert!(entities.len() >= 2, "Should find multiple entities");
}

#[test]
fn test_neural_indian_entities() {
    let ner = create_neural_ner();

    let text = "Ratan Tata visited Infosys headquarters in Bangalore";
    let entities = ner.extract(text).expect("Extraction failed");

    println!("Text: {}", text);
    println!("Entities found:");
    for e in &entities {
        println!(
            "  - {} ({:?}, confidence: {:.2})",
            e.text, e.entity_type, e.confidence
        );
    }

    assert!(!entities.is_empty(), "Should extract Indian entities");
}

#[test]
fn test_neural_latency() {
    let ner = create_neural_ner();

    // Warm up
    let _ = ner.extract("Warmup text");

    let text = "Microsoft CEO Satya Nadella visited Google in Mountain View, California";

    let start = std::time::Instant::now();
    let iterations = 10;

    for _ in 0..iterations {
        let _ = ner.extract(text).expect("Extraction failed");
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("Average neural NER latency: {:.2}ms", avg_ms);
    println!("Expected: 5-50ms on modern hardware");

    // Neural model should complete in reasonable time (under 100ms per extraction)
    assert!(
        avg_ms < 100.0,
        "Neural NER should be under 100ms, got {:.2}ms",
        avg_ms
    );
}

#[test]
fn test_neural_vs_fallback_comparison() {
    let neural_ner = create_neural_ner();
    let fallback_ner = {
        let config = NerConfig::default();
        NeuralNer::new_fallback(config)
    };

    let text = "Tim Cook from Apple met Sundar Pichai from Google in San Francisco";

    let neural_entities = neural_ner.extract(text).expect("Neural extraction failed");
    let fallback_entities = fallback_ner
        .extract(text)
        .expect("Fallback extraction failed");

    println!("Text: {}", text);
    println!("\nNeural entities ({}):", neural_entities.len());
    for e in &neural_entities {
        println!("  - {} ({:?}, {:.2})", e.text, e.entity_type, e.confidence);
    }

    println!("\nFallback entities ({}):", fallback_entities.len());
    for e in &fallback_entities {
        println!("  - {} ({:?}, {:.2})", e.text, e.entity_type, e.confidence);
    }

    // Both should find entities
    assert!(!neural_entities.is_empty(), "Neural should find entities");
    assert!(
        !fallback_entities.is_empty(),
        "Fallback should find entities"
    );
}

#[test]
fn test_neural_long_text() {
    let ner = create_neural_ner();

    let text = "In a landmark development for India's technology sector, \
                Mukesh Ambani's Reliance Industries announced a strategic partnership \
                with Google and Facebook. The collaboration will see Jio Platforms \
                expanding its 5G infrastructure across Mumbai, Delhi, and Bangalore. \
                Industry experts including Nandan Nilekani praised the move.";

    let entities = ner.extract(text).expect("Extraction failed");

    println!("Long text extraction:");
    println!("Text length: {} chars", text.len());
    println!("Entities found ({}):", entities.len());
    for e in &entities {
        println!("  - {} ({:?}, {:.2})", e.text, e.entity_type, e.confidence);
    }

    // Should find multiple entities in long text
    assert!(
        entities.len() >= 3,
        "Should find multiple entities in long text"
    );
}

#[test]
fn test_neural_edge_cases() {
    let ner = create_neural_ner();

    // Empty text
    let empty_entities = ner.extract("").expect("Empty extraction failed");
    assert!(
        empty_entities.is_empty(),
        "Empty text should have no entities"
    );

    // No entities
    let no_entity_text = "the quick brown fox jumps over the lazy dog";
    let no_entities = ner
        .extract(no_entity_text)
        .expect("No entity extraction failed");
    println!("No entity text: {} entities found", no_entities.len());

    // Unicode text
    let unicode_text = "Microsoft ने नया प्रोडक्ट लॉन्च किया";
    let unicode_entities = ner
        .extract(unicode_text)
        .expect("Unicode extraction failed");
    println!("Unicode text: {} entities found", unicode_entities.len());
    for e in &unicode_entities {
        println!("  - {} ({:?})", e.text, e.entity_type);
    }
}
