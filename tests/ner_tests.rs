//! Integration tests for Neural NER module
//!
//! Tests the NER module with various real-world scenarios:
//! - Organization extraction (Indian-first, then global)
//! - Location extraction (Indian cities, global cities)
//! - Person name extraction
//! - Mixed entity extraction
//! - Edge cases and stress tests

use shodh_memory::embeddings::ner::{NerConfig, NerEntityType, NeuralNer};
use std::path::PathBuf;

/// Create a fallback NER instance for testing
fn create_test_ner() -> NeuralNer {
    let config = NerConfig {
        model_path: PathBuf::from("nonexistent.onnx"),
        tokenizer_path: PathBuf::from("nonexistent.json"),
        max_length: 128,
        confidence_threshold: 0.5,
    };
    NeuralNer::new_fallback(config)
}

// ==================== Indian Entity Tests ====================

mod indian_entities {
    use super::*;

    #[test]
    fn test_indian_it_companies() {
        let ner = create_test_ner();
        let companies = vec![
            "TCS", "Infosys", "Wipro", "HCL", "Flipkart", "Paytm", "Zomato", "Swiggy", "Ola",
        ];

        for company in companies {
            let text = format!("{} reported strong growth this quarter", company);
            let entities = ner.extract(&text).unwrap();
            let found = entities.iter().any(|e| {
                e.text.to_lowercase() == company.to_lowercase()
                    && e.entity_type == NerEntityType::Organization
            });
            assert!(
                found
                    || entities
                        .iter()
                        .any(|e| e.text.to_lowercase() == company.to_lowercase()),
                "Should find Indian IT company: {}",
                company
            );
        }
    }

    #[test]
    fn test_indian_conglomerates() {
        let ner = create_test_ner();
        let conglomerates = vec!["Tata", "Reliance", "Adani", "Birla", "Mahindra", "Godrej"];

        for company in conglomerates {
            let text = format!("{} group announced expansion plans", company);
            let entities = ner.extract(&text).unwrap();
            assert!(
                entities.iter().any(|e| e.text == company),
                "Should find Indian conglomerate: {}",
                company
            );
        }
    }

    #[test]
    fn test_indian_metro_cities() {
        let ner = create_test_ner();
        let metros = vec![
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Bengaluru",
            "Hyderabad",
            "Chennai",
            "Kolkata",
            "Pune",
        ];

        for city in metros {
            let text = format!("The office is located in {}", city);
            let entities = ner.extract(&text).unwrap();
            let found = entities
                .iter()
                .find(|e| e.text.to_lowercase() == city.to_lowercase());
            assert!(found.is_some(), "Should find Indian metro: {}", city);
            assert_eq!(
                found.unwrap().entity_type,
                NerEntityType::Location,
                "{} should be Location",
                city
            );
        }
    }

    #[test]
    fn test_indian_tier2_cities() {
        let ner = create_test_ner();
        let cities = vec![
            "Jaipur",
            "Lucknow",
            "Kochi",
            "Coimbatore",
            "Indore",
            "Nagpur",
            "Vadodara",
        ];

        for city in cities {
            let text = format!("New branch opening in {}", city);
            let entities = ner.extract(&text).unwrap();
            let found = entities
                .iter()
                .find(|e| e.text.to_lowercase() == city.to_lowercase());
            assert!(found.is_some(), "Should find Indian tier-2 city: {}", city);
        }
    }

    #[test]
    fn test_indian_tech_hubs() {
        let ner = create_test_ner();
        let text = "The startup ecosystem in Bangalore and Hyderabad is thriving";
        let entities = ner.extract(text).unwrap();

        // Should find both tech hubs
        let bangalore = entities
            .iter()
            .any(|e| e.text == "Bangalore" && e.entity_type == NerEntityType::Location);
        let hyderabad = entities
            .iter()
            .any(|e| e.text == "Hyderabad" && e.entity_type == NerEntityType::Location);

        assert!(bangalore, "Should find Bangalore");
        assert!(hyderabad, "Should find Hyderabad");
    }
}

// ==================== Global Entity Tests ====================

mod global_entities {
    use super::*;

    #[test]
    fn test_global_tech_giants() {
        let ner = create_test_ner();
        let companies = vec![
            "Microsoft",
            "Google",
            "Apple",
            "Amazon",
            "Meta",
            "Netflix",
            "Tesla",
        ];

        for company in companies {
            let text = format!("{} announced new products", company);
            let entities = ner.extract(&text).unwrap();
            let found = entities.iter().find(|e| e.text == company);
            assert!(found.is_some(), "Should find global tech: {}", company);
            assert_eq!(
                found.unwrap().entity_type,
                NerEntityType::Organization,
                "{} should be Organization",
                company
            );
        }
    }

    #[test]
    fn test_global_cities() {
        let ner = create_test_ner();
        let cities = vec![
            "London",
            "Tokyo",
            "Singapore",
            "Dubai",
            "Sydney",
            "Berlin",
            "Paris",
        ];

        for city in cities {
            let text = format!("Conference will be held in {}", city);
            let entities = ner.extract(&text).unwrap();
            let found = entities
                .iter()
                .find(|e| e.text.to_lowercase() == city.to_lowercase());
            assert!(found.is_some(), "Should find global city: {}", city);
        }
    }
}

// ==================== Mixed Entity Tests ====================

mod mixed_entities {
    use super::*;

    #[test]
    fn test_org_and_location_same_sentence() {
        let ner = create_test_ner();
        let text = "Microsoft headquarters in Seattle employs thousands";
        let entities = ner.extract(text).unwrap();

        let has_microsoft = entities
            .iter()
            .any(|e| e.text == "Microsoft" && e.entity_type == NerEntityType::Organization);
        let has_seattle = entities
            .iter()
            .any(|e| e.text == "Seattle" && e.entity_type == NerEntityType::Location);

        assert!(has_microsoft, "Should find Microsoft as Organization");
        assert!(has_seattle, "Should find Seattle as Location");
    }

    #[test]
    fn test_indian_company_global_location() {
        let ner = create_test_ner();
        let text = "Infosys opened new office in London";
        let entities = ner.extract(text).unwrap();

        let has_infosys = entities
            .iter()
            .any(|e| e.text == "Infosys" && e.entity_type == NerEntityType::Organization);
        let has_london = entities
            .iter()
            .any(|e| e.text == "London" && e.entity_type == NerEntityType::Location);

        assert!(has_infosys, "Should find Infosys");
        assert!(has_london, "Should find London");
    }

    #[test]
    fn test_global_company_indian_location() {
        let ner = create_test_ner();
        let text = "Google expanding operations in Bangalore";
        let entities = ner.extract(text).unwrap();

        let has_google = entities
            .iter()
            .any(|e| e.text == "Google" && e.entity_type == NerEntityType::Organization);
        let has_bangalore = entities
            .iter()
            .any(|e| e.text == "Bangalore" && e.entity_type == NerEntityType::Location);

        assert!(has_google, "Should find Google");
        assert!(has_bangalore, "Should find Bangalore");
    }

    #[test]
    fn test_multiple_organizations() {
        let ner = create_test_ner();
        let text = "Microsoft and Google are competing with Apple";
        let entities = ner.extract(text).unwrap();

        let org_count = entities
            .iter()
            .filter(|e| e.entity_type == NerEntityType::Organization)
            .count();

        assert!(org_count >= 2, "Should find at least 2 organizations");
    }

    #[test]
    fn test_multiple_locations() {
        let ner = create_test_ner();
        let text = "Traveling from Mumbai to Delhi via Bangalore";
        let entities = ner.extract(text).unwrap();

        let loc_count = entities
            .iter()
            .filter(|e| e.entity_type == NerEntityType::Location)
            .count();

        assert!(loc_count >= 2, "Should find at least 2 locations");
    }
}

// ==================== Edge Case Tests ====================

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_input() {
        let ner = create_test_ner();
        let entities = ner.extract("").unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let ner = create_test_ner();
        let entities = ner.extract("   \t\n   ").unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_no_entities() {
        let ner = create_test_ner();
        // Fallback regex NER may extract keywords as Misc entities from common text.
        // Verify that a simple sentence produces no Person/Org/Location entities.
        let text = "it was cold and dark";
        let entities = ner.extract(text).unwrap();
        let named_entities: Vec<_> = entities
            .iter()
            .filter(|e| {
                matches!(
                    e.entity_type,
                    NerEntityType::Person | NerEntityType::Organization | NerEntityType::Location
                )
            })
            .collect();
        assert!(
            named_entities.is_empty(),
            "expected no named entities from: {}",
            text
        );
    }

    #[test]
    fn test_unicode_text() {
        let ner = create_test_ner();
        // Hindi text with English entity
        let text = "Microsoft ने नया प्रोडक्ट लॉन्च किया";
        let entities = ner.extract(text).unwrap();
        // Should still find Microsoft
        assert!(entities.iter().any(|e| e.text == "Microsoft"));
    }

    #[test]
    fn test_very_long_text() {
        let ner = create_test_ner();
        // Generate long text with entities sprinkled throughout
        let mut text = String::new();
        for i in 0..100 {
            if i % 10 == 0 {
                text.push_str("Microsoft is great. ");
            } else {
                text.push_str("This is some filler text. ");
            }
        }

        let entities = ner.extract(&text).unwrap();
        // Should find Microsoft (deduplicated)
        let microsoft_count = entities.iter().filter(|e| e.text == "Microsoft").count();
        assert_eq!(microsoft_count, 1, "Should deduplicate Microsoft mentions");
    }

    #[test]
    fn test_special_characters() {
        let ner = create_test_ner();
        let text = "Microsoft™ vs Google® in Seattle!";
        let entities = ner.extract(text).unwrap();

        // Should extract entities without special characters
        for entity in &entities {
            assert!(!entity.text.contains('™'));
            assert!(!entity.text.contains('®'));
        }
    }

    #[test]
    fn test_case_sensitivity() {
        let ner = create_test_ner();

        // Test lowercase - should not be found (not capitalized)
        let entities_lower = ner.extract("microsoft google apple").unwrap();
        // Lowercase words typically won't be detected as entities

        // Test proper case - should be found
        let entities_proper = ner.extract("Microsoft Google Apple").unwrap();
        assert!(
            !entities_proper.is_empty(),
            "Properly capitalized entities should be found"
        );
    }

    #[test]
    fn test_numbers_in_text() {
        let ner = create_test_ner();
        let text = "Microsoft earned $100 billion in 2023 in Seattle";
        let entities = ner.extract(text).unwrap();

        // Should still find Microsoft and Seattle
        assert!(entities.iter().any(|e| e.text == "Microsoft"));
        assert!(entities.iter().any(|e| e.text == "Seattle"));
    }

    #[test]
    fn test_abbreviations() {
        let ner = create_test_ner();
        let text = "TCS and IBM are competing";
        let entities = ner.extract(text).unwrap();

        // TCS should be found (Indian company)
        let has_tcs = entities
            .iter()
            .any(|e| e.text.to_lowercase() == "tcs" || e.text == "TCS");
        assert!(has_tcs, "Should find TCS abbreviation");
    }
}

// ==================== Performance Tests ====================

mod performance {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_extraction_latency_short_text() {
        let ner = create_test_ner();
        let text = "Microsoft is headquartered in Seattle";

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _ = ner.extract(text).unwrap();
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        println!("Average extraction time (short): {:.3}ms", avg_ms);
        // Fallback should be very fast - under 1ms
        assert!(
            avg_ms < 1.0,
            "Fallback extraction should be under 1ms, got {:.3}ms",
            avg_ms
        );
    }

    #[test]
    fn test_extraction_latency_long_text() {
        let ner = create_test_ner();
        let text =
            "Microsoft and Google are expanding their operations in Bangalore and Hyderabad. \
                   Amazon is also opening new offices in Mumbai. Meanwhile, Flipkart and Paytm \
                   continue to grow in the Indian market. The tech industry in India is booming \
                   with cities like Chennai, Pune, and Kolkata becoming new tech hubs.";

        let start = Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let _ = ner.extract(text).unwrap();
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        println!("Average extraction time (long): {:.3}ms", avg_ms);
        // Fallback should still be fast - under 5ms for long text
        assert!(
            avg_ms < 5.0,
            "Fallback extraction should be under 5ms, got {:.3}ms",
            avg_ms
        );
    }

    #[test]
    fn test_many_entities() {
        let ner = create_test_ner();

        // Text with many entities
        let text = "Microsoft Google Apple Amazon Meta Netflix Tesla IBM Oracle SAP \
                   Infosys TCS Wipro Flipkart Paytm Zomato Swiggy Ola \
                   Mumbai Delhi Bangalore Hyderabad Chennai Kolkata Pune \
                   London Tokyo Singapore Dubai Sydney Berlin Paris";

        let start = Instant::now();
        let entities = ner.extract(text).unwrap();
        let elapsed = start.elapsed();

        println!(
            "Extracted {} entities in {:.3}ms",
            entities.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        assert!(entities.len() >= 10, "Should find many entities");
        assert!(
            elapsed.as_millis() < 20,
            "Should complete in under 20ms even with many entities"
        );
    }
}

// ==================== Accuracy Tests ====================

mod accuracy {
    use super::*;

    /// Test dataset: (text, expected_entities)
    fn get_test_cases() -> Vec<(&'static str, Vec<(&'static str, NerEntityType)>)> {
        vec![
            (
                "Microsoft CEO visited Seattle",
                vec![
                    ("Microsoft", NerEntityType::Organization),
                    ("Seattle", NerEntityType::Location),
                ],
            ),
            (
                "Infosys opened office in Bangalore",
                vec![
                    ("Infosys", NerEntityType::Organization),
                    ("Bangalore", NerEntityType::Location),
                ],
            ),
            (
                "Google and Amazon compete globally",
                vec![
                    ("Google", NerEntityType::Organization),
                    ("Amazon", NerEntityType::Organization),
                ],
            ),
            (
                "Mumbai to Delhi flight",
                vec![
                    ("Mumbai", NerEntityType::Location),
                    ("Delhi", NerEntityType::Location),
                ],
            ),
        ]
    }

    #[test]
    fn test_accuracy_on_test_set() {
        let ner = create_test_ner();
        let test_cases = get_test_cases();

        let mut total_expected = 0;
        let mut total_found = 0;

        for (text, expected) in test_cases {
            let entities = ner.extract(text).unwrap();
            total_expected += expected.len();

            for (expected_text, expected_type) in expected {
                let found = entities.iter().any(|e| {
                    e.text.to_lowercase() == expected_text.to_lowercase()
                        && e.entity_type == expected_type
                });
                if found {
                    total_found += 1;
                } else {
                    println!(
                        "Missing: {} ({:?}) in '{}'",
                        expected_text, expected_type, text
                    );
                }
            }
        }

        let recall = total_found as f64 / total_expected as f64;
        println!(
            "Recall: {:.1}% ({}/{})",
            recall * 100.0,
            total_found,
            total_expected
        );

        // Fallback should achieve at least 70% recall on known entities
        assert!(
            recall >= 0.7,
            "Recall should be at least 70%, got {:.1}%",
            recall * 100.0
        );
    }
}
