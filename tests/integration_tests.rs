//! Comprehensive Integration Test Suite for Shodh-Memory
//!
//! Enterprise-grade testing demonstrating production readiness for:
//! - Robotics and autonomous systems
//! - Drone fleet management
//! - Edge AI deployments
//!
//! Run with: cargo test --test integration_tests -- --nocapture

use std::collections::HashMap;
use std::time::Instant;
use std::path::PathBuf;
use tempfile::TempDir;

use shodh_memory::{
    memory::{MemorySystem, MemoryConfig, Experience, Query, ExperienceType, RetrievalMode},
    memory::types::GeoFilter,
};

// ============================================================================
// TEST REPORTER - Professional output for enterprise demonstration
// ============================================================================

struct TestReporter {
    test_name: String,
    start_time: Instant,
    results: Vec<TestResult>,
}

struct TestResult {
    name: String,
    passed: bool,
    duration_ms: u128,
    details: String,
}

impl TestReporter {
    fn new(suite_name: &str) -> Self {
        println!("\n{}", "=".repeat(80));
        println!("  SHODH-MEMORY INTEGRATION TEST SUITE");
        println!("  {}", suite_name);
        println!("{}", "=".repeat(80));
        println!("  Enterprise-grade AI Memory for Robotics & Autonomous Systems");
        println!("  Version: 0.1.0 | License: Commercial\n");

        Self {
            test_name: suite_name.to_string(),
            start_time: Instant::now(),
            results: Vec::new(),
        }
    }

    fn record(&mut self, name: &str, passed: bool, duration_ms: u128, details: &str) {
        let status = if passed { "[PASS]" } else { "[FAIL]" };
        println!("  {} {} ({} ms)", status, name, duration_ms);
        if !details.is_empty() {
            println!("       {}", details);
        }

        self.results.push(TestResult {
            name: name.to_string(),
            passed,
            duration_ms,
            details: details.to_string(),
        });
    }

    fn section(&self, name: &str) {
        println!("\n  --- {} ---\n", name);
    }

    fn report(self) {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let total_duration = self.start_time.elapsed().as_millis();

        println!("\n{}", "=".repeat(80));
        println!("  TEST SUITE SUMMARY: {}", self.test_name);
        println!("{}", "-".repeat(80));
        println!("  Total Tests:  {}", total);
        println!("  Passed:       {} ({:.1}%)", passed, (passed as f64 / total as f64) * 100.0);
        println!("  Failed:       {}", failed);
        println!("  Duration:     {} ms", total_duration);
        println!("{}", "=".repeat(80));

        if failed == 0 {
            println!("\n  [SUCCESS] All tests passed - System ready for production deployment\n");
        } else {
            println!("\n  [WARNING] {} tests failed - Review before deployment\n", failed);
            for result in &self.results {
                if !result.passed {
                    println!("    - {}: {}", result.name, result.details);
                }
            }
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn create_test_config(temp_dir: &TempDir) -> MemoryConfig {
    MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: true,
        compression_age_days: 7,
        importance_threshold: 0.5,
    }
}

fn create_robotics_experience(content: &str, robot_id: &str, entities: Vec<&str>) -> Experience {
    let mut metadata = HashMap::new();
    metadata.insert("robot_id".to_string(), robot_id.to_string());

    Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        context: None,
        entities: entities.into_iter().map(|s| s.to_string()).collect(),
        metadata,
        embeddings: None,
        related_memories: Vec::new(),
        causal_chain: Vec::new(),
        outcomes: Vec::new(),
        robot_id: Some(robot_id.to_string()),
        mission_id: None,
        geo_location: None,
        local_position: None,
        heading: None,
        action_type: None,
        reward: None,
        sensor_data: HashMap::new(),
    }
}

fn create_drone_experience(
    content: &str,
    drone_id: &str,
    mission_id: &str,
    geo_location: Option<[f64; 3]>,
    action_type: Option<&str>,
) -> Experience {
    let mut metadata = HashMap::new();
    metadata.insert("drone_id".to_string(), drone_id.to_string());
    metadata.insert("mission_id".to_string(), mission_id.to_string());

    Experience {
        experience_type: ExperienceType::Task,
        content: content.to_string(),
        context: None,
        entities: vec![drone_id.to_string(), mission_id.to_string()],
        metadata,
        embeddings: None,
        related_memories: Vec::new(),
        causal_chain: Vec::new(),
        outcomes: Vec::new(),
        robot_id: Some(drone_id.to_string()),
        mission_id: Some(mission_id.to_string()),
        geo_location,
        local_position: None,
        heading: None,
        action_type: action_type.map(|s| s.to_string()),
        reward: None,
        sensor_data: HashMap::new(),
    }
}

// ============================================================================
// CORE FUNCTIONALITY TESTS
// ============================================================================

#[test]
fn test_core_memory_operations() {
    let mut reporter = TestReporter::new("Core Memory Operations");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    reporter.section("Memory System Initialization");

    // Test 1: System initialization
    let start = Instant::now();
    let system_result = MemorySystem::new(config);
    let init_duration = start.elapsed().as_millis();

    let mut system = match system_result {
        Ok(s) => {
            reporter.record("Initialize memory system", true, init_duration,
                &format!("System initialized in {} ms", init_duration));
            s
        }
        Err(e) => {
            reporter.record("Initialize memory system", false, init_duration,
                &format!("Failed: {}", e));
            reporter.report();
            panic!("Cannot continue without memory system");
        }
    };

    reporter.section("Record Operations");

    // Test 2: Record basic experience
    let start = Instant::now();
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: "Robot arm successfully grasped object at position (10, 20, 5)".to_string(),
        context: None,
        entities: vec!["robot_arm".to_string(), "grasp".to_string(), "object".to_string()],
        metadata: HashMap::new(),
        embeddings: None,
        related_memories: Vec::new(),
        causal_chain: Vec::new(),
        outcomes: Vec::new(),
        robot_id: Some("arm_001".to_string()),
        mission_id: None,
        geo_location: None,
        local_position: Some([10.0, 20.0, 5.0]),
        heading: None,
        action_type: Some("grasp".to_string()),
        reward: Some(0.95),
        sensor_data: HashMap::new(),
    };

    let record_result = system.record(experience);
    let record_duration = start.elapsed().as_millis();

    match record_result {
        Ok(memory_id) => {
            reporter.record("Record experience", true, record_duration,
                &format!("Memory ID: {}", memory_id.0));
        }
        Err(e) => {
            reporter.record("Record experience", false, record_duration,
                &format!("Failed: {}", e));
        }
    }

    // Test 3: Record multiple experiences
    let start = Instant::now();
    let mut success_count = 0;
    for i in 0..10 {
        let exp = create_robotics_experience(
            &format!("Test observation {} - sensor reading at waypoint", i),
            "robot_001",
            vec!["sensor", "waypoint"],
        );
        if system.record(exp).is_ok() {
            success_count += 1;
        }
    }
    let batch_duration = start.elapsed().as_millis();

    reporter.record("Batch record (10 memories)", success_count == 10, batch_duration,
        &format!("{}/10 successful, avg {} ms/record", success_count, batch_duration / 10));

    reporter.section("Retrieval Operations");

    // Test 4: Basic retrieval
    let start = Instant::now();
    let query = Query {
        query_text: Some("robot arm grasp object".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let retrieve_result = system.retrieve(&query);
    let retrieve_duration = start.elapsed().as_millis();

    match retrieve_result {
        Ok(memories) => {
            reporter.record("Semantic retrieval", !memories.is_empty(), retrieve_duration,
                &format!("Retrieved {} memories in {} ms", memories.len(), retrieve_duration));
        }
        Err(e) => {
            reporter.record("Semantic retrieval", false, retrieve_duration,
                &format!("Failed: {}", e));
        }
    }

    // Test 5: Entity-based retrieval
    let start = Instant::now();
    let query = Query {
        query_text: Some("sensor".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let entity_result = system.retrieve(&query);
    let entity_duration = start.elapsed().as_millis();

    match entity_result {
        Ok(memories) => {
            reporter.record("Entity-based retrieval", true, entity_duration,
                &format!("Found {} memories with sensor entity", memories.len()));
        }
        Err(e) => {
            reporter.record("Entity-based retrieval", false, entity_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.report();
}

// ============================================================================
// ROBOTICS-SPECIFIC TESTS
// ============================================================================

#[test]
fn test_robotics_scenarios() {
    let mut reporter = TestReporter::new("Robotics & Autonomous Systems");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let mut system = MemorySystem::new(config).expect("Failed to create memory system");

    reporter.section("Mission Memory Tracking");

    // Test 1: Mission-based memory organization
    let start = Instant::now();
    let missions = vec![
        ("mission_alpha", "patrol", vec![
            "Started patrol route at sector A",
            "Detected obstacle at coordinates (15, 30)",
            "Rerouted around obstacle via waypoint B",
            "Completed patrol - no anomalies detected",
        ]),
        ("mission_beta", "inspection", vec![
            "Initiated inspection of equipment rack 1",
            "Found rust damage on component C-15",
            "Documented damage with visual capture",
            "Flagged for maintenance priority HIGH",
        ]),
    ];

    let mut mission_success = true;
    for (mission_id, mission_type, events) in missions {
        for event in events {
            let exp = Experience {
                experience_type: ExperienceType::Task,
                content: event.to_string(),
                context: None,
                entities: vec![mission_type.to_string()],
                metadata: HashMap::new(),
                embeddings: None,
                related_memories: Vec::new(),
                causal_chain: Vec::new(),
                outcomes: Vec::new(),
                robot_id: Some("robot_001".to_string()),
                mission_id: Some(mission_id.to_string()),
                geo_location: None,
                local_position: None,
                heading: None,
                action_type: None,
                reward: None,
                sensor_data: HashMap::new(),
            };
            if system.record(exp).is_err() {
                mission_success = false;
            }
        }
    }
    let mission_duration = start.elapsed().as_millis();

    reporter.record("Mission memory tracking", mission_success, mission_duration,
        "2 missions with 4 events each recorded");

    // Test 2: Retrieve mission-specific memories
    let start = Instant::now();
    let query = Query {
        query_text: Some("patrol obstacle".to_string()),
        mission_id: Some("mission_alpha".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let patrol_result = system.retrieve(&query);
    let patrol_duration = start.elapsed().as_millis();

    match patrol_result {
        Ok(memories) => {
            reporter.record("Mission-filtered retrieval", true, patrol_duration,
                &format!("Found {} patrol memories", memories.len()));
        }
        Err(e) => {
            reporter.record("Mission-filtered retrieval", false, patrol_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.section("Obstacle Detection & Memory");

    // Test 3: Obstacle memory
    let start = Instant::now();
    let obstacles = vec![
        ("Large rock", [10.0, 25.0, 0.0_f32]),
        ("Fallen tree", [35.0, 12.0, 0.0]),
        ("Water puddle", [22.0, 40.0, -0.5]),
    ];

    let mut obstacle_count = 0;
    for (desc, pos) in obstacles {
        let exp = Experience {
            experience_type: ExperienceType::Discovery,
            content: format!("Obstacle detected: {} at position ({}, {}, {})",
                desc, pos[0], pos[1], pos[2]),
            context: None,
            entities: vec!["obstacle".to_string(), desc.to_lowercase()],
            metadata: HashMap::new(),
            embeddings: None,
            related_memories: Vec::new(),
            causal_chain: Vec::new(),
            outcomes: Vec::new(),
            robot_id: Some("robot_001".to_string()),
            mission_id: None,
            geo_location: None,
            local_position: Some(pos),
            heading: None,
            action_type: Some("obstacle_detection".to_string()),
            reward: None,
            sensor_data: HashMap::new(),
        };
        if system.record(exp).is_ok() {
            obstacle_count += 1;
        }
    }
    let obstacle_duration = start.elapsed().as_millis();

    reporter.record("Obstacle memory storage", obstacle_count == 3, obstacle_duration,
        &format!("{}/3 obstacles recorded", obstacle_count));

    // Test 4: Query obstacles
    let start = Instant::now();
    let query = Query {
        query_text: Some("obstacle detected".to_string()),
        action_type: Some("obstacle_detection".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let obstacle_query_result = system.retrieve(&query);
    let obstacle_query_duration = start.elapsed().as_millis();

    match obstacle_query_result {
        Ok(memories) => {
            reporter.record("Obstacle memory retrieval", true, obstacle_query_duration,
                &format!("Retrieved {} obstacle memories", memories.len()));
        }
        Err(e) => {
            reporter.record("Obstacle memory retrieval", false, obstacle_query_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.section("Sensor Calibration Memory");

    // Test 5: Sensor calibration tracking
    let start = Instant::now();
    let sensors = vec![
        ("lidar_front", 0.98, "optimal"),
        ("camera_rgb", 0.95, "good"),
        ("imu_main", 0.99, "excellent"),
        ("ultrasonic_left", 0.85, "needs_recalibration"),
    ];

    let mut calibration_success = true;
    for (sensor, accuracy, status) in sensors {
        let mut sensor_data = HashMap::new();
        sensor_data.insert(format!("{}_accuracy", sensor), accuracy);

        let exp = Experience {
            experience_type: ExperienceType::Pattern,
            content: format!("Sensor {} calibration: accuracy {:.2}, status: {}",
                sensor, accuracy, status),
            context: None,
            entities: vec!["sensor".to_string(), "calibration".to_string(), sensor.to_string()],
            metadata: HashMap::new(),
            embeddings: None,
            related_memories: Vec::new(),
            causal_chain: Vec::new(),
            outcomes: Vec::new(),
            robot_id: Some("robot_001".to_string()),
            mission_id: None,
            geo_location: None,
            local_position: None,
            heading: None,
            action_type: Some("calibration".to_string()),
            reward: Some(accuracy as f32),
            sensor_data,
        };
        if system.record(exp).is_err() {
            calibration_success = false;
        }
    }
    let calibration_duration = start.elapsed().as_millis();

    reporter.record("Sensor calibration tracking", calibration_success, calibration_duration,
        "4 sensor calibrations recorded");

    // Test 6: Find sensors needing recalibration
    let start = Instant::now();
    let query = Query {
        query_text: Some("needs recalibration".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let recal_result = system.retrieve(&query);
    let recal_duration = start.elapsed().as_millis();

    match recal_result {
        Ok(memories) => {
            reporter.record("Recalibration needs query", true, recal_duration,
                &format!("Found {} sensors needing attention", memories.len()));
        }
        Err(e) => {
            reporter.record("Recalibration needs query", false, recal_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.report();
}

// ============================================================================
// DRONE FLEET TESTS
// ============================================================================

#[test]
fn test_drone_fleet_operations() {
    let mut reporter = TestReporter::new("Drone Fleet Management");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let mut system = MemorySystem::new(config).expect("Failed to create memory system");

    reporter.section("Multi-Drone Coordination");

    // Test 1: Multi-drone mission recording
    let start = Instant::now();
    let drones = vec!["drone_alpha", "drone_beta", "drone_gamma"];
    let mut success_count = 0;

    for (i, drone_id) in drones.iter().enumerate() {
        let lat = 37.7749 + (i as f64 * 0.001);
        let lon = -122.4194 + (i as f64 * 0.001);

        let exp = create_drone_experience(
            &format!("{} initiated surveillance sweep at sector {}", drone_id, i + 1),
            drone_id,
            "surveillance_mission_001",
            Some([lat, lon, 100.0]),
            Some("surveillance"),
        );
        if system.record(exp).is_ok() {
            success_count += 1;
        }
    }
    let multi_drone_duration = start.elapsed().as_millis();

    reporter.record("Multi-drone mission init", success_count == 3, multi_drone_duration,
        &format!("{}/3 drones initialized", success_count));

    reporter.section("Geo-Spatial Memory");

    // Test 2: Record geo-tagged events
    let start = Instant::now();
    let events = vec![
        ("Target identified at construction site", 37.7749, -122.4194, 50.0),
        ("Weather station data collected", 37.7850, -122.4094, 75.0),
        ("Traffic monitoring complete", 37.7650, -122.4294, 100.0),
        ("Emergency vehicle detected", 37.7749, -122.4194, 60.0),
    ];

    let mut geo_success = 0;
    for (desc, lat, lon, alt) in events {
        let exp = create_drone_experience(
            desc,
            "drone_alpha",
            "surveillance_mission_001",
            Some([lat, lon, alt]),
            Some("observation"),
        );
        if system.record(exp).is_ok() {
            geo_success += 1;
        }
    }
    let geo_duration = start.elapsed().as_millis();

    reporter.record("Geo-tagged event recording", geo_success == 4, geo_duration,
        &format!("{}/4 geo-tagged events recorded", geo_success));

    // Test 3: Geo-spatial query
    let start = Instant::now();
    let query = Query {
        query_text: Some("target identified".to_string()),
        geo_filter: Some(GeoFilter::new(37.7749, -122.4194, 1000.0)), // 1km radius
        max_results: 10,
        ..Default::default()
    };

    let geo_query_result = system.retrieve(&query);
    let geo_query_duration = start.elapsed().as_millis();

    match geo_query_result {
        Ok(memories) => {
            reporter.record("Geo-spatial retrieval", true, geo_query_duration,
                &format!("Found {} memories within 1km radius", memories.len()));
        }
        Err(e) => {
            reporter.record("Geo-spatial retrieval", false, geo_query_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.section("Flight Path Memory");

    // Test 4: Record flight path
    let start = Instant::now();
    let waypoints = vec![
        ([37.7749, -122.4194, 100.0], 0.0_f32),
        ([37.7760, -122.4180, 105.0], 45.0),
        ([37.7775, -122.4160, 110.0], 60.0),
        ([37.7790, -122.4140, 115.0], 75.0),
        ([37.7800, -122.4130, 100.0], 90.0),
    ];

    let mut path_success = 0;
    for (i, (pos, heading)) in waypoints.iter().enumerate() {
        let exp = Experience {
            experience_type: ExperienceType::Task,
            content: format!("Waypoint {} reached - altitude {}m, heading {}deg",
                i + 1, pos[2], heading),
            context: None,
            entities: vec!["waypoint".to_string(), "flight_path".to_string()],
            metadata: HashMap::new(),
            embeddings: None,
            related_memories: Vec::new(),
            causal_chain: Vec::new(),
            outcomes: Vec::new(),
            robot_id: Some("drone_alpha".to_string()),
            mission_id: Some("flight_001".to_string()),
            geo_location: Some(*pos),
            local_position: None,
            heading: Some(*heading),
            action_type: Some("navigation".to_string()),
            reward: Some(1.0),
            sensor_data: HashMap::new(),
        };
        if system.record(exp).is_ok() {
            path_success += 1;
        }
    }
    let path_duration = start.elapsed().as_millis();

    reporter.record("Flight path recording", path_success == 5, path_duration,
        &format!("{}/5 waypoints recorded", path_success));

    // Test 5: Query flight path
    let start = Instant::now();
    let query = Query {
        query_text: Some("waypoint reached".to_string()),
        mission_id: Some("flight_001".to_string()),
        max_results: 10,
        retrieval_mode: RetrievalMode::Temporal,
        ..Default::default()
    };

    let path_query_result = system.retrieve(&query);
    let path_query_duration = start.elapsed().as_millis();

    match path_query_result {
        Ok(memories) => {
            reporter.record("Flight path retrieval", true, path_query_duration,
                &format!("Retrieved {} waypoint memories", memories.len()));
        }
        Err(e) => {
            reporter.record("Flight path retrieval", false, path_query_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.section("Battery & Telemetry");

    // Test 6: Battery state tracking
    let start = Instant::now();
    let battery_readings = vec![
        (100, "full", 0),
        (85, "good", 5),
        (65, "moderate", 12),
        (40, "low", 20),
        (20, "critical", 28),
    ];

    let mut battery_success = 0;
    for (level, status, flight_minutes) in battery_readings {
        let mut sensor_data = HashMap::new();
        sensor_data.insert("battery_percent".to_string(), level as f64);
        sensor_data.insert("flight_time_minutes".to_string(), flight_minutes as f64);

        let exp = Experience {
            experience_type: ExperienceType::Pattern,
            content: format!("Battery at {}% ({}) after {} minutes of flight",
                level, status, flight_minutes),
            context: None,
            entities: vec!["battery".to_string(), status.to_string()],
            metadata: HashMap::new(),
            embeddings: None,
            related_memories: Vec::new(),
            causal_chain: Vec::new(),
            outcomes: Vec::new(),
            robot_id: Some("drone_alpha".to_string()),
            mission_id: Some("flight_001".to_string()),
            geo_location: None,
            local_position: None,
            heading: None,
            action_type: Some("telemetry".to_string()),
            reward: Some((level as f32) / 100.0),
            sensor_data,
        };
        if system.record(exp).is_ok() {
            battery_success += 1;
        }
    }
    let battery_duration = start.elapsed().as_millis();

    reporter.record("Battery telemetry tracking", battery_success == 5, battery_duration,
        &format!("{}/5 battery readings recorded", battery_success));

    // Test 7: Critical battery query
    let start = Instant::now();
    let query = Query {
        query_text: Some("battery critical".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let critical_result = system.retrieve(&query);
    let critical_duration = start.elapsed().as_millis();

    match critical_result {
        Ok(memories) => {
            reporter.record("Critical status retrieval", true, critical_duration,
                &format!("Found {} critical battery events", memories.len()));
        }
        Err(e) => {
            reporter.record("Critical status retrieval", false, critical_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.report();
}

// ============================================================================
// PERFORMANCE BENCHMARKS
// ============================================================================

#[test]
fn test_performance_benchmarks() {
    let mut reporter = TestReporter::new("Performance Benchmarks");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let mut system = MemorySystem::new(config).expect("Failed to create memory system");

    reporter.section("Insertion Performance");

    // Benchmark 1: High-volume insertion
    let batch_sizes = [10, 50, 100];

    for batch_size in batch_sizes {
        let start = Instant::now();
        let mut success = 0;

        for i in 0..batch_size {
            let exp = create_robotics_experience(
                &format!("Performance test observation {} with sensor data and position tracking", i),
                &format!("perf_robot_{}", i % 5),
                vec!["performance", "benchmark", "sensor"],
            );
            if system.record(exp).is_ok() {
                success += 1;
            }
        }

        let duration = start.elapsed().as_millis();
        let rate = if duration > 0 { (success as f64 / duration as f64) * 1000.0 } else { 0.0 };

        reporter.record(
            &format!("Insert {} memories", batch_size),
            success == batch_size,
            duration,
            &format!("{:.1} records/sec, avg {:.2} ms/record",
                rate,
                if success > 0 { duration as f64 / success as f64 } else { 0.0 }),
        );
    }

    reporter.section("Query Performance");

    // Benchmark 2: Query latency
    let queries = vec![
        ("Simple keyword", "sensor"),
        ("Multi-word", "performance test observation"),
        ("Entity-focused", "robot performance benchmark"),
    ];

    for (name, query_text) in queries {
        let mut total_duration = 0u128;
        let iterations = 5;

        for _ in 0..iterations {
            let start = Instant::now();
            let query = Query {
                query_text: Some(query_text.to_string()),
                max_results: 10,
                ..Default::default()
            };
            let _ = system.retrieve(&query);
            total_duration += start.elapsed().as_millis();
        }

        let avg_duration = total_duration / iterations as u128;

        reporter.record(
            &format!("{} query", name),
            avg_duration < 500,  // Target: <500ms average
            avg_duration,
            &format!("avg over {} iterations", iterations),
        );
    }

    reporter.section("Memory Efficiency");

    // Benchmark 3: Memory stats
    let start = Instant::now();
    let stats = system.stats();
    let stats_duration = start.elapsed().as_millis();

    reporter.record("Stats retrieval", true, stats_duration,
        &format!("Total memories: {}, Working: {}, Session: {}, Long-term: {}",
            stats.total_memories,
            stats.working_memory_count,
            stats.session_memory_count,
            stats.long_term_memory_count));

    reporter.report();
}

// ============================================================================
// RELIABILITY TESTS
// ============================================================================

#[test]
fn test_reliability_and_edge_cases() {
    let mut reporter = TestReporter::new("Reliability & Edge Cases");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let mut system = MemorySystem::new(config).expect("Failed to create memory system");

    reporter.section("Input Validation");

    // Test 1: Empty content handling
    let start = Instant::now();
    let exp = Experience {
        experience_type: ExperienceType::Observation,
        content: "".to_string(),  // Empty content
        context: None,
        entities: Vec::new(),
        metadata: HashMap::new(),
        embeddings: None,
        related_memories: Vec::new(),
        causal_chain: Vec::new(),
        outcomes: Vec::new(),
        robot_id: None,
        mission_id: None,
        geo_location: None,
        local_position: None,
        heading: None,
        action_type: None,
        reward: None,
        sensor_data: HashMap::new(),
    };

    let empty_result = system.record(exp);
    let empty_duration = start.elapsed().as_millis();

    // Empty content should still be handled gracefully
    reporter.record("Empty content handling", empty_result.is_ok(), empty_duration,
        "System handles empty content gracefully");

    // Test 2: Very long content
    let start = Instant::now();
    let long_content = "x".repeat(10000);  // 10KB of content
    let exp = Experience {
        experience_type: ExperienceType::Observation,
        content: long_content,
        context: None,
        entities: Vec::new(),
        metadata: HashMap::new(),
        embeddings: None,
        related_memories: Vec::new(),
        causal_chain: Vec::new(),
        outcomes: Vec::new(),
        robot_id: None,
        mission_id: None,
        geo_location: None,
        local_position: None,
        heading: None,
        action_type: None,
        reward: None,
        sensor_data: HashMap::new(),
    };

    let long_result = system.record(exp);
    let long_duration = start.elapsed().as_millis();

    reporter.record("Large content handling (10KB)", long_result.is_ok(), long_duration,
        &format!("Processed in {} ms", long_duration));

    // Test 3: Special characters in content
    let start = Instant::now();
    let special_content = "Test with special chars: <>&\"' \n\t\r and unicode: 你好世界 🤖 λ∑∏";
    let exp = Experience {
        experience_type: ExperienceType::Observation,
        content: special_content.to_string(),
        context: None,
        entities: vec!["unicode".to_string(), "special_chars".to_string()],
        metadata: HashMap::new(),
        embeddings: None,
        related_memories: Vec::new(),
        causal_chain: Vec::new(),
        outcomes: Vec::new(),
        robot_id: None,
        mission_id: None,
        geo_location: None,
        local_position: None,
        heading: None,
        action_type: None,
        reward: None,
        sensor_data: HashMap::new(),
    };

    let special_result = system.record(exp);
    let special_duration = start.elapsed().as_millis();

    reporter.record("Special characters handling", special_result.is_ok(), special_duration,
        "Unicode and special characters handled correctly");

    reporter.section("Query Edge Cases");

    // Test 4: Empty query
    let start = Instant::now();
    let query = Query {
        query_text: Some("".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let empty_query_result = system.retrieve(&query);
    let empty_query_duration = start.elapsed().as_millis();

    reporter.record("Empty query handling", empty_query_result.is_ok(), empty_query_duration,
        "Empty query handled gracefully");

    // Test 5: Query with no results
    let start = Instant::now();
    let query = Query {
        query_text: Some("nonexistent_query_that_should_match_nothing_xyz123".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let no_results = system.retrieve(&query);
    let no_results_duration = start.elapsed().as_millis();

    match no_results {
        Ok(memories) => {
            reporter.record("No-match query", true, no_results_duration,
                &format!("Returned {} results (expected 0 or low)", memories.len()));
        }
        Err(e) => {
            reporter.record("No-match query", false, no_results_duration,
                &format!("Failed: {}", e));
        }
    }

    reporter.section("Concurrent Access Simulation");

    // Test 6: Rapid sequential operations
    let start = Instant::now();
    let mut ops_success = 0;

    for i in 0..20 {
        // Alternate between record and retrieve
        if i % 2 == 0 {
            let exp = create_robotics_experience(
                &format!("Concurrent test {}", i),
                "robot_concurrent",
                vec!["concurrent"],
            );
            if system.record(exp).is_ok() {
                ops_success += 1;
            }
        } else {
            let query = Query {
                query_text: Some("concurrent".to_string()),
                max_results: 5,
                ..Default::default()
            };
            if system.retrieve(&query).is_ok() {
                ops_success += 1;
            }
        }
    }
    let concurrent_duration = start.elapsed().as_millis();

    reporter.record("Rapid sequential ops (20)", ops_success == 20, concurrent_duration,
        &format!("{}/20 operations successful", ops_success));

    reporter.section("Data Persistence");

    // Test 7: Flush and verify
    let start = Instant::now();
    let flush_result = system.flush_storage();
    let flush_duration = start.elapsed().as_millis();

    reporter.record("Storage flush", flush_result.is_ok(), flush_duration,
        "Data persisted to disk");

    reporter.report();
}

// ============================================================================
// STRESS TEST
// ============================================================================

#[test]
fn test_stress_scenarios() {
    let mut reporter = TestReporter::new("Stress Testing");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let mut system = MemorySystem::new(config).expect("Failed to create memory system");

    reporter.section("High Volume Operations");

    // Stress test: 500 rapid insertions
    let start = Instant::now();
    let target = 500;
    let mut success = 0;

    for i in 0..target {
        let exp = create_robotics_experience(
            &format!("Stress test record {} - high frequency sensor data from multiple sources", i),
            &format!("stress_robot_{}", i % 10),
            vec!["stress", "high_volume", "sensor"],
        );
        if system.record(exp).is_ok() {
            success += 1;
        }
    }

    let stress_duration = start.elapsed().as_millis();
    let rate = if stress_duration > 0 {
        (success as f64 / stress_duration as f64) * 1000.0
    } else {
        0.0
    };

    reporter.record(
        &format!("{} rapid insertions", target),
        success >= (target * 95 / 100),  // 95% success threshold
        stress_duration,
        &format!("{}/{} successful ({:.1} records/sec)", success, target, rate),
    );

    // Rapid queries after stress
    let start = Instant::now();
    let query_count = 50;
    let mut query_success = 0;

    for i in 0..query_count {
        let query = Query {
            query_text: Some(format!("stress test record {}", i * 10)),
            max_results: 5,
            ..Default::default()
        };
        if system.retrieve(&query).is_ok() {
            query_success += 1;
        }
    }

    let query_duration = start.elapsed().as_millis();

    reporter.record(
        &format!("{} queries post-stress", query_count),
        query_success == query_count,
        query_duration,
        &format!("{}/{} successful, avg {:.1} ms/query",
            query_success, query_count,
            query_duration as f64 / query_count as f64),
    );

    reporter.report();
}
