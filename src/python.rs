// Python bindings for shodh-memory using PyO3
//
// Comprehensive robotics memory system with:
// - Position/GeoLocation tracking
// - Decision tree learning (action → outcome → reward)
// - Sensor data patterns and anomaly detection
// - Environmental context (weather, terrain, lighting)
// - Failure tracking and recovery patterns
// - Learned behaviors and predictions

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::memory::types::{
    Experience, ExperienceType, ForgetCriteria, GeoFilter, Memory, MemoryId,
};
use crate::memory::{MemoryConfig, MemorySystem, Query, RetrievalMode};
use chrono::{DateTime, Utc};

// ============================================================================
// Position - Local coordinates (x, y, z in meters)
// ============================================================================

/// Local position in robot's coordinate frame (meters)
#[pyclass(name = "Position")]
#[derive(Clone, Debug)]
pub struct PyPosition {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub z: f32,
}

#[pymethods]
impl PyPosition {
    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, z=0.0))]
    fn new(x: f32, y: f32, z: f32) -> Self {
        PyPosition { x, y, z }
    }

    fn distance_to(&self, other: &PyPosition) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    fn to_list(&self) -> Vec<f32> {
        vec![self.x, self.y, self.z]
    }

    fn __repr__(&self) -> String {
        format!("Position(x={}, y={}, z={})", self.x, self.y, self.z)
    }
}

// ============================================================================
// GeoLocation - GPS coordinates
// ============================================================================

/// GPS coordinates (WGS84) for outdoor/drone navigation
#[pyclass(name = "GeoLocation")]
#[derive(Clone, Debug)]
pub struct PyGeoLocation {
    #[pyo3(get, set)]
    pub latitude: f64,
    #[pyo3(get, set)]
    pub longitude: f64,
    #[pyo3(get, set)]
    pub altitude: f64,
}

#[pymethods]
impl PyGeoLocation {
    #[new]
    #[pyo3(signature = (latitude=0.0, longitude=0.0, altitude=0.0))]
    fn new(latitude: f64, longitude: f64, altitude: f64) -> Self {
        PyGeoLocation {
            latitude,
            longitude,
            altitude,
        }
    }

    fn to_list(&self) -> Vec<f64> {
        vec![self.latitude, self.longitude, self.altitude]
    }

    fn __repr__(&self) -> String {
        format!(
            "GeoLocation(lat={}, lon={}, alt={})",
            self.latitude, self.longitude, self.altitude
        )
    }
}

// ============================================================================
// GeoFilter - Spatial query filter
// ============================================================================

/// Filter memories by geographic radius
#[pyclass(name = "GeoFilter")]
#[derive(Clone, Debug)]
pub struct PyGeoFilter {
    #[pyo3(get, set)]
    pub latitude: f64,
    #[pyo3(get, set)]
    pub longitude: f64,
    #[pyo3(get, set)]
    pub radius_meters: f64,
}

#[pymethods]
impl PyGeoFilter {
    #[new]
    #[pyo3(signature = (latitude, longitude, radius_meters))]
    fn new(latitude: f64, longitude: f64, radius_meters: f64) -> Self {
        PyGeoFilter {
            latitude,
            longitude,
            radius_meters,
        }
    }

    #[staticmethod]
    fn from_center(center: &PyGeoLocation, radius_meters: f64) -> Self {
        PyGeoFilter {
            latitude: center.latitude,
            longitude: center.longitude,
            radius_meters,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeoFilter(lat={}, lon={}, radius={}m)",
            self.latitude, self.longitude, self.radius_meters
        )
    }
}

// ============================================================================
// DecisionContext - For action-outcome learning
// ============================================================================

/// Context for a decision - what conditions led to this action?
#[pyclass(name = "DecisionContext")]
#[derive(Clone, Debug, Default)]
pub struct PyDecisionContext {
    /// State variables: {"battery_low": "true", "obstacle_ahead": "true"}
    #[pyo3(get, set)]
    pub state: HashMap<String, String>,
    /// Action parameters: {"speed": "0.5", "turn_angle": "45"}
    #[pyo3(get, set)]
    pub action_params: HashMap<String, String>,
    /// Confidence in this decision (0.0-1.0)
    #[pyo3(get, set)]
    pub confidence: Option<f32>,
    /// Alternative actions that were considered
    #[pyo3(get, set)]
    pub alternatives: Vec<String>,
}

#[pymethods]
impl PyDecisionContext {
    #[new]
    #[pyo3(signature = (state=None, action_params=None, confidence=None, alternatives=None))]
    fn new(
        state: Option<HashMap<String, String>>,
        action_params: Option<HashMap<String, String>>,
        confidence: Option<f32>,
        alternatives: Option<Vec<String>>,
    ) -> Self {
        PyDecisionContext {
            state: state.unwrap_or_default(),
            action_params: action_params.unwrap_or_default(),
            confidence,
            alternatives: alternatives.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DecisionContext(state={:?}, confidence={:?})",
            self.state, self.confidence
        )
    }
}

// ============================================================================
// Outcome - Result of an action
// ============================================================================

/// Outcome of an action for learning
#[pyclass(name = "Outcome")]
#[derive(Clone, Debug, Default)]
pub struct PyOutcome {
    /// success, failure, partial, aborted, timeout
    #[pyo3(get, set)]
    pub outcome_type: String,
    /// Detailed description
    #[pyo3(get, set)]
    pub details: Option<String>,
    /// Reward signal for RL (-1.0 to 1.0)
    #[pyo3(get, set)]
    pub reward: Option<f32>,
    /// Was this predicted correctly?
    #[pyo3(get, set)]
    pub prediction_accurate: Option<bool>,
}

#[pymethods]
impl PyOutcome {
    #[new]
    #[pyo3(signature = (outcome_type, details=None, reward=None, prediction_accurate=None))]
    fn new(
        outcome_type: String,
        details: Option<String>,
        reward: Option<f32>,
        prediction_accurate: Option<bool>,
    ) -> Self {
        PyOutcome {
            outcome_type,
            details,
            reward,
            prediction_accurate,
        }
    }

    fn is_success(&self) -> bool {
        self.outcome_type == "success"
    }

    fn is_failure(&self) -> bool {
        self.outcome_type == "failure"
    }

    fn __repr__(&self) -> String {
        format!(
            "Outcome(type={}, reward={:?})",
            self.outcome_type, self.reward
        )
    }
}

// ============================================================================
// Environment - Environmental context
// ============================================================================

/// Environmental conditions during operation
#[pyclass(name = "Environment")]
#[derive(Clone, Debug, Default)]
pub struct PyEnvironment {
    /// Weather: {"wind_speed": "15", "visibility": "good"}
    #[pyo3(get, set)]
    pub weather: HashMap<String, String>,
    /// indoor, outdoor, urban, rural, water, aerial
    #[pyo3(get, set)]
    pub terrain_type: Option<String>,
    /// bright, dim, dark, variable
    #[pyo3(get, set)]
    pub lighting: Option<String>,
    /// Other agents: [{"id": "drone_002", "distance": "50m"}]
    #[pyo3(get, set)]
    pub nearby_agents: Vec<HashMap<String, String>>,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    #[pyo3(signature = (weather=None, terrain_type=None, lighting=None, nearby_agents=None))]
    fn new(
        weather: Option<HashMap<String, String>>,
        terrain_type: Option<String>,
        lighting: Option<String>,
        nearby_agents: Option<Vec<HashMap<String, String>>>,
    ) -> Self {
        PyEnvironment {
            weather: weather.unwrap_or_default(),
            terrain_type,
            lighting,
            nearby_agents: nearby_agents.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Environment(terrain={:?}, lighting={:?})",
            self.terrain_type, self.lighting
        )
    }
}

// ============================================================================
// MemorySystem - Main interface
// ============================================================================

/// Python wrapper for MemorySystem with comprehensive robotics support
#[pyclass(name = "MemorySystem")]
pub struct PyMemorySystem {
    inner: MemorySystem,
    robot_id: Option<String>,
    mission_id: Option<String>,
}

#[pymethods]
impl PyMemorySystem {
    /// Create a new memory system
    #[new]
    #[pyo3(signature = (storage_path=None, robot_id=None))]
    fn new(storage_path: Option<String>, robot_id: Option<String>) -> PyResult<Self> {
        let path = storage_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("./shodh_data"));

        let config = MemoryConfig {
            storage_path: path,
            ..Default::default()
        };

        let inner = MemorySystem::new(config).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create memory system: {}", e))
        })?;

        Ok(PyMemorySystem {
            inner,
            robot_id,
            mission_id: None,
        })
    }

    // === Mission Management ===

    fn start_mission(&mut self, mission_id: String) {
        self.mission_id = Some(mission_id);
    }

    fn end_mission(&mut self) {
        self.mission_id = None;
    }

    fn current_mission(&self) -> Option<String> {
        self.mission_id.clone()
    }

    // === Core Memory API (Unified with HTTP client) ===

    /// Store a memory with full robotics and decision-making support
    ///
    /// This is the primary API for storing memories, matching the HTTP client's remember() method.
    #[pyo3(signature = (
        content,
        memory_type="observation",
        position=None,
        geo_location=None,
        heading=None,
        action_type=None,
        sensor_data=None,
        decision_context=None,
        outcome=None,
        environment=None,
        is_failure=false,
        is_anomaly=false,
        severity=None,
        recovery_action=None,
        root_cause=None,
        pattern_id=None,
        predicted_outcome=None,
        tags=None,
        entities=None,
        metadata=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn remember(
        &mut self,
        content: String,
        memory_type: &str,
        position: Option<&PyPosition>,
        geo_location: Option<&PyGeoLocation>,
        heading: Option<f32>,
        action_type: Option<String>,
        sensor_data: Option<HashMap<String, f64>>,
        decision_context: Option<&PyDecisionContext>,
        outcome: Option<&PyOutcome>,
        environment: Option<&PyEnvironment>,
        is_failure: bool,
        is_anomaly: bool,
        severity: Option<String>,
        recovery_action: Option<String>,
        root_cause: Option<String>,
        pattern_id: Option<String>,
        predicted_outcome: Option<String>,
        tags: Option<Vec<String>>,
        entities: Option<Vec<String>>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<String> {
        let exp_type = match memory_type.to_lowercase().as_str() {
            "observation" | "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "discovery" => ExperienceType::Discovery,
            "error" => ExperienceType::Error,
            "pattern" | "learning" => ExperienceType::Learning,
            "decision" => ExperienceType::Decision,
            "conversation" => ExperienceType::Conversation,
            _ => ExperienceType::Context,
        };

        let experience = Experience {
            experience_type: exp_type,
            content,
            context: None,
            entities: entities.unwrap_or_default(),
            metadata: metadata.unwrap_or_default(),
            embeddings: None,
            related_memories: vec![],
            causal_chain: vec![],
            outcomes: outcome
                .as_ref()
                .and_then(|o| o.details.clone())
                .map(|d| vec![d])
                .unwrap_or_default(),
            // Robotics fields
            robot_id: self.robot_id.clone(),
            mission_id: self.mission_id.clone(),
            geo_location: geo_location.map(|g| [g.latitude, g.longitude, g.altitude]),
            local_position: position.map(|p| [p.x, p.y, p.z]),
            heading,
            action_type,
            reward: outcome.as_ref().and_then(|o| o.reward),
            sensor_data: sensor_data.unwrap_or_default(),
            // Decision fields
            decision_context: decision_context.map(|d| d.state.clone()),
            action_params: decision_context.map(|d| d.action_params.clone()),
            outcome_type: outcome.map(|o| o.outcome_type.clone()),
            outcome_details: outcome.and_then(|o| o.details.clone()),
            confidence: decision_context.and_then(|d| d.confidence),
            alternatives_considered: decision_context
                .map(|d| d.alternatives.clone())
                .unwrap_or_default(),
            // Environment fields
            weather: environment.map(|e| e.weather.clone()),
            terrain_type: environment.and_then(|e| e.terrain_type.clone()),
            lighting: environment.and_then(|e| e.lighting.clone()),
            nearby_agents: environment
                .map(|e| e.nearby_agents.clone())
                .unwrap_or_default(),
            // Failure fields
            is_failure,
            is_anomaly,
            severity,
            recovery_action,
            root_cause,
            // Pattern fields
            pattern_id,
            predicted_outcome,
            prediction_accurate: outcome.and_then(|o| o.prediction_accurate),
            tags: tags.unwrap_or_default(),
            // Multimodal fields (not exposed in Python API yet)
            image_embeddings: None,
            audio_embeddings: None,
            video_embeddings: None,
            media_refs: vec![],
            // Temporal extraction
            temporal_refs: vec![],
        };

        let memory_id = self
            .inner
            .remember(experience, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to record memory: {}", e)))?;

        Ok(memory_id.0.to_string())
    }

    // === Convenience Methods for Common Robotics Operations ===

    /// Record a decision with context, action, and outcome
    #[pyo3(signature = (description, action_type, decision_context, outcome, position=None, geo_location=None, sensor_data=None))]
    fn record_decision(
        &mut self,
        description: String,
        action_type: String,
        decision_context: &PyDecisionContext,
        outcome: &PyOutcome,
        position: Option<&PyPosition>,
        geo_location: Option<&PyGeoLocation>,
        sensor_data: Option<HashMap<String, f64>>,
    ) -> PyResult<String> {
        self.remember(
            description,
            "decision",
            position,
            geo_location,
            None,
            Some(action_type),
            sensor_data,
            Some(decision_context),
            Some(outcome),
            None,
            outcome.is_failure(),
            false,
            None,
            None,
            None,
            None,
            None,
            Some(vec!["decision".to_string()]),
            None,
            None,
        )
    }

    /// Record a failure event with recovery information
    #[pyo3(signature = (description, severity, root_cause=None, recovery_action=None, position=None, sensor_data=None))]
    fn record_failure(
        &mut self,
        description: String,
        severity: String,
        root_cause: Option<String>,
        recovery_action: Option<String>,
        position: Option<&PyPosition>,
        sensor_data: Option<HashMap<String, f64>>,
    ) -> PyResult<String> {
        self.remember(
            description,
            "error",
            position,
            None,
            None,
            None,
            sensor_data,
            None,
            None,
            None,
            true,
            false,
            Some(severity),
            recovery_action,
            root_cause,
            None,
            None,
            Some(vec!["failure".to_string()]),
            None,
            None,
        )
    }

    /// Record an anomaly detection
    #[pyo3(signature = (description, sensor_data, severity="warning", position=None))]
    fn record_anomaly(
        &mut self,
        description: String,
        sensor_data: HashMap<String, f64>,
        severity: &str,
        position: Option<&PyPosition>,
    ) -> PyResult<String> {
        self.remember(
            description,
            "discovery",
            position,
            None,
            None,
            None,
            Some(sensor_data),
            None,
            None,
            None,
            false,
            true,
            Some(severity.to_string()),
            None,
            None,
            None,
            None,
            Some(vec!["anomaly".to_string()]),
            None,
            None,
        )
    }

    /// Record sensor readings with pattern detection
    #[pyo3(signature = (sensor_name, readings, pattern_id=None, is_anomaly=false, position=None))]
    fn record_sensor(
        &mut self,
        sensor_name: String,
        readings: HashMap<String, f64>,
        pattern_id: Option<String>,
        is_anomaly: bool,
        position: Option<&PyPosition>,
    ) -> PyResult<String> {
        let mut tags = vec!["sensor".to_string(), sensor_name.clone()];
        if is_anomaly {
            tags.push("anomaly".to_string());
        }

        self.remember(
            format!("Sensor {}: {:?}", sensor_name, readings),
            if is_anomaly { "error" } else { "observation" },
            position,
            None,
            None,
            Some("sensor_reading".to_string()),
            Some(readings),
            None,
            None,
            None,
            false,
            is_anomaly,
            None,
            None,
            None,
            pattern_id,
            None,
            Some(tags),
            Some(vec![sensor_name]),
            None,
        )
    }

    /// Record an obstacle detection
    #[pyo3(signature = (description, distance=None, confidence=None, position=None, geo_location=None))]
    fn record_obstacle(
        &mut self,
        description: String,
        distance: Option<f64>,
        confidence: Option<f64>,
        position: Option<&PyPosition>,
        geo_location: Option<&PyGeoLocation>,
    ) -> PyResult<String> {
        let mut sensor_data = HashMap::new();
        if let Some(d) = distance {
            sensor_data.insert("distance".to_string(), d);
        }
        if let Some(c) = confidence {
            sensor_data.insert("confidence".to_string(), c);
        }

        self.remember(
            format!("Obstacle detected: {}", description),
            "discovery",
            position,
            geo_location,
            None,
            Some("obstacle_detection".to_string()),
            Some(sensor_data),
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            Some(vec!["obstacle".to_string()]),
            Some(vec!["obstacle".to_string(), description.to_lowercase()]),
            None,
        )
    }

    /// Record a waypoint event
    #[pyo3(signature = (waypoint_id, status="reached", position=None, geo_location=None))]
    fn record_waypoint(
        &mut self,
        waypoint_id: String,
        status: &str,
        position: Option<&PyPosition>,
        geo_location: Option<&PyGeoLocation>,
    ) -> PyResult<String> {
        self.remember(
            format!("Waypoint {}: {}", waypoint_id, status),
            "task",
            position,
            geo_location,
            None,
            Some("navigation".to_string()),
            None,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            Some(vec!["waypoint".to_string(), "navigation".to_string()]),
            Some(vec!["waypoint".to_string(), waypoint_id]),
            None,
        )
    }

    // === Query Methods (Unified with HTTP client) ===

    /// Search and retrieve memories with comprehensive filtering
    ///
    /// This is the primary API for retrieving memories, matching the HTTP client's recall() method.
    #[pyo3(signature = (
        query,
        limit=10,
        mode="hybrid",
        mission_id=None,
        action_type=None,
        geo_filter=None,
        min_importance=None,
        outcome_type=None,
        failures_only=false,
        anomalies_only=false,
        severity=None,
        tags=None,
        pattern_id=None,
        terrain_type=None,
        min_confidence=None,
        max_confidence=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn recall(
        &self,
        query: String,
        limit: usize,
        mode: &str,
        mission_id: Option<String>,
        action_type: Option<String>,
        geo_filter: Option<&PyGeoFilter>,
        min_importance: Option<f32>,
        outcome_type: Option<String>,
        failures_only: bool,
        anomalies_only: bool,
        severity: Option<String>,
        tags: Option<Vec<String>>,
        pattern_id: Option<String>,
        terrain_type: Option<String>,
        min_confidence: Option<f32>,
        max_confidence: Option<f32>,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let retrieval_mode = match mode.to_lowercase().as_str() {
            "semantic" | "similarity" => RetrievalMode::Similarity,
            "temporal" => RetrievalMode::Temporal,
            "hybrid" => RetrievalMode::Hybrid,
            "causal" => RetrievalMode::Causal,
            "associative" => RetrievalMode::Associative,
            "spatial" => RetrievalMode::Spatial,
            "mission" => RetrievalMode::Mission,
            "action_outcome" => RetrievalMode::ActionOutcome,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown retrieval mode: {}. Valid modes: semantic, temporal, hybrid, causal, associative, spatial, mission, action_outcome",
                    mode
                )))
            }
        };

        let confidence_range = match (min_confidence, max_confidence) {
            (Some(min), Some(max)) => Some((min, max)),
            (Some(min), None) => Some((min, 1.0)),
            (None, Some(max)) => Some((0.0, max)),
            (None, None) => None,
        };

        let query_obj = Query {
            user_id: None,
            query_text: Some(query),
            query_embedding: None,
            time_range: None,
            experience_types: None,
            importance_threshold: min_importance,
            robot_id: self.robot_id.clone(),
            mission_id,
            geo_filter: geo_filter
                .map(|f| GeoFilter::new(f.latitude, f.longitude, f.radius_meters)),
            action_type,
            reward_range: None,
            outcome_type,
            failures_only,
            anomalies_only,
            severity,
            tags,
            pattern_id,
            terrain_type,
            confidence_range,
            prospective_signals: None,
            episode_id: None,
            max_results: limit,
            retrieval_mode,
            offset: 0,
        };

        let memories = self
            .inner
            .recall(&query_obj)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to recall memories: {}", e)))?;

        Python::with_gil(|py| {
            memories
                .iter()
                .map(|mem| memory_to_dict(py, mem))
                .collect::<PyResult<Vec<_>>>()
        })
    }

    /// Find similar situations for decision-making
    #[pyo3(signature = (action_type, decision_context=None, max_results=10))]
    fn find_similar_decisions(
        &self,
        action_type: String,
        decision_context: Option<&PyDecisionContext>,
        max_results: usize,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let query_text = if let Some(ctx) = decision_context {
            format!("action {} with context {:?}", action_type, ctx.state)
        } else {
            format!("action {}", action_type)
        };

        self.recall(
            query_text,
            max_results,
            "action_outcome",
            None,
            Some(action_type),
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Find all failures for a given action type
    #[pyo3(signature = (action_type=None, severity=None, max_results=20))]
    fn find_failures(
        &self,
        action_type: Option<String>,
        severity: Option<String>,
        max_results: usize,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        self.recall(
            "failure error problem".to_string(),
            max_results,
            "hybrid",
            None,
            action_type,
            None,
            None,
            Some("failure".to_string()),
            true,
            false,
            severity,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Find anomalies in sensor data
    #[pyo3(signature = (sensor_name=None, max_results=20))]
    fn find_anomalies(
        &self,
        sensor_name: Option<String>,
        max_results: usize,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let tags = sensor_name.map(|s| vec![s, "anomaly".to_string()]);
        self.recall(
            "anomaly unusual unexpected".to_string(),
            max_results,
            "hybrid",
            None,
            None,
            None,
            None,
            None,
            false,
            true,
            None,
            tags,
            None,
            None,
            None,
            None,
        )
    }

    /// Find memories matching a learned pattern
    #[pyo3(signature = (pattern_id, max_results=20))]
    fn find_by_pattern(
        &self,
        pattern_id: String,
        max_results: usize,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        self.recall(
            format!("pattern {}", pattern_id),
            max_results,
            "hybrid",
            None,
            None,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            Some(pattern_id),
            None,
            None,
            None,
        )
    }

    // === Statistics ===

    fn get_stats(&self) -> PyResult<HashMap<String, usize>> {
        let stats = self.inner.stats();

        let mut result = HashMap::new();
        result.insert("total_memories".to_string(), stats.total_memories);
        result.insert(
            "working_memory_count".to_string(),
            stats.working_memory_count,
        );
        result.insert(
            "session_memory_count".to_string(),
            stats.session_memory_count,
        );
        result.insert(
            "long_term_memory_count".to_string(),
            stats.long_term_memory_count,
        );
        result.insert("compressed_count".to_string(), stats.compressed_count);
        result.insert(
            "promotions_to_session".to_string(),
            stats.promotions_to_session,
        );
        result.insert(
            "promotions_to_longterm".to_string(),
            stats.promotions_to_longterm,
        );
        result.insert("total_retrievals".to_string(), stats.total_retrievals);

        Ok(result)
    }

    fn flush(&self) -> PyResult<()> {
        self.inner
            .flush_storage()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to flush: {}", e)))
    }

    // === Context & Introspection API (Matching REST) ===

    /// Get categorized context summary for session bootstrap
    ///
    /// Returns decisions, learnings, patterns, errors organized for LLM consumption.
    /// Matches REST /api/context_summary endpoint.
    #[pyo3(signature = (max_items=5, include_decisions=true, include_learnings=true, include_context=true))]
    fn context_summary(
        &self,
        max_items: usize,
        include_decisions: bool,
        include_learnings: bool,
        include_context: bool,
    ) -> PyResult<HashMap<String, PyObject>> {
        let all_memories = self
            .inner
            .get_all_memories()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get memories: {}", e)))?;

        let total_memories = all_memories.len();

        // Categorize memories by type
        let mut decisions: Vec<(String, String, f32, String)> = Vec::new();
        let mut learnings: Vec<(String, String, f32, String)> = Vec::new();
        let mut context: Vec<(String, String, f32, String)> = Vec::new();
        let mut patterns: Vec<(String, String, f32, String)> = Vec::new();
        let mut errors: Vec<(String, String, f32, String)> = Vec::new();

        for m in all_memories {
            let item = (
                m.id.0.to_string(),
                m.experience.content.chars().take(200).collect(),
                m.importance(),
                m.created_at.to_rfc3339(),
            );

            match m.experience.experience_type {
                ExperienceType::Decision => decisions.push(item),
                ExperienceType::Learning => learnings.push(item),
                ExperienceType::Context | ExperienceType::Observation => context.push(item),
                ExperienceType::Pattern => patterns.push(item),
                ExperienceType::Error => errors.push(item),
                _ => context.push(item),
            }
        }

        // Sort by importance and truncate
        fn sort_and_truncate(
            mut items: Vec<(String, String, f32, String)>,
            max: usize,
        ) -> Vec<(String, String, f32, String)> {
            items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(max);
            items
        }

        Python::with_gil(|py| {
            let mut result = HashMap::new();

            result.insert("total_memories".to_string(), total_memories.into_py(py));

            // Convert items to list of dicts
            let to_py_list = |items: Vec<(String, String, f32, String)>| -> PyObject {
                let list: Vec<HashMap<String, PyObject>> = items
                    .into_iter()
                    .map(|(id, content, importance, created_at)| {
                        let mut item = HashMap::new();
                        item.insert("id".to_string(), id.into_py(py));
                        item.insert("content".to_string(), content.into_py(py));
                        item.insert("importance".to_string(), importance.into_py(py));
                        item.insert("created_at".to_string(), created_at.into_py(py));
                        item
                    })
                    .collect();
                list.into_py(py)
            };

            result.insert(
                "decisions".to_string(),
                if include_decisions {
                    to_py_list(sort_and_truncate(decisions, max_items))
                } else {
                    Vec::<HashMap<String, PyObject>>::new().into_py(py)
                },
            );

            result.insert(
                "learnings".to_string(),
                if include_learnings {
                    to_py_list(sort_and_truncate(learnings, max_items))
                } else {
                    Vec::<HashMap<String, PyObject>>::new().into_py(py)
                },
            );

            result.insert(
                "context".to_string(),
                if include_context {
                    to_py_list(sort_and_truncate(context, max_items))
                } else {
                    Vec::<HashMap<String, PyObject>>::new().into_py(py)
                },
            );

            result.insert(
                "patterns".to_string(),
                to_py_list(sort_and_truncate(patterns, max_items)),
            );

            result.insert(
                "errors".to_string(),
                to_py_list(sort_and_truncate(errors, 3.min(max_items))),
            );

            Ok(result)
        })
    }

    /// List all memories
    ///
    /// Matches REST /api/list/{user_id} endpoint.
    #[pyo3(signature = (limit=None, memory_type=None))]
    fn list_memories(
        &self,
        limit: Option<usize>,
        memory_type: Option<&str>,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let all_memories = self
            .inner
            .get_all_memories()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get memories: {}", e)))?;

        // Filter by type if specified
        let filtered: Vec<_> = if let Some(type_filter) = memory_type {
            let type_lower = type_filter.to_lowercase();
            all_memories
                .into_iter()
                .filter(|m| {
                    let mem_type = format!("{:?}", m.experience.experience_type).to_lowercase();
                    mem_type == type_lower
                })
                .collect()
        } else {
            all_memories
        };

        // Apply limit
        let limited: Vec<_> = if let Some(lim) = limit {
            filtered.into_iter().take(lim).collect()
        } else {
            filtered
        };

        Python::with_gil(|py| {
            limited
                .iter()
                .map(|mem| memory_to_dict(py, mem))
                .collect::<PyResult<Vec<_>>>()
        })
    }

    /// Get a single memory by ID
    ///
    /// Matches REST /api/memory/{id} GET endpoint.
    fn get_memory(&self, memory_id: &str) -> PyResult<HashMap<String, PyObject>> {
        let id = MemoryId(
            uuid::Uuid::parse_str(memory_id)
                .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?,
        );

        let memory = self
            .inner
            .get_memory(&id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get memory: {}", e)))?;

        Python::with_gil(|py| memory_to_dict(py, &memory))
    }

    /// Search memories by tags (no embedding needed)
    ///
    /// Matches REST /api/recall/tags endpoint.
    #[pyo3(signature = (tags, limit=20))]
    fn recall_by_tags(
        &self,
        tags: Vec<String>,
        limit: usize,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let all_memories = self
            .inner
            .get_all_memories()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get memories: {}", e)))?;

        // Filter by tags - memory must have ANY of the provided tags
        let filtered: Vec<_> = all_memories
            .into_iter()
            .filter(|m| {
                m.experience.tags.iter().any(|t| {
                    tags.iter()
                        .any(|search_tag| t.eq_ignore_ascii_case(search_tag))
                })
            })
            .take(limit)
            .collect();

        Python::with_gil(|py| {
            filtered
                .iter()
                .map(|mem| memory_to_dict(py, mem))
                .collect::<PyResult<Vec<_>>>()
        })
    }

    /// Search memories by date range
    ///
    /// Matches REST /api/recall/date endpoint.
    /// Dates should be ISO 8601 format (e.g., "2024-01-01T00:00:00Z")
    #[pyo3(signature = (start, end, limit=20))]
    fn recall_by_date(
        &self,
        start: &str,
        end: &str,
        limit: usize,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let start_dt = chrono::DateTime::parse_from_rfc3339(start)
            .map_err(|e| PyValueError::new_err(format!("Invalid start date: {}", e)))?
            .with_timezone(&chrono::Utc);
        let end_dt = chrono::DateTime::parse_from_rfc3339(end)
            .map_err(|e| PyValueError::new_err(format!("Invalid end date: {}", e)))?
            .with_timezone(&chrono::Utc);

        let all_memories = self
            .inner
            .get_all_memories()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get memories: {}", e)))?;

        let filtered: Vec<_> = all_memories
            .into_iter()
            .filter(|m| m.created_at >= start_dt && m.created_at <= end_dt)
            .take(limit)
            .collect();

        Python::with_gil(|py| {
            filtered
                .iter()
                .map(|mem| memory_to_dict(py, mem))
                .collect::<PyResult<Vec<_>>>()
        })
    }

    /// Get knowledge graph statistics
    ///
    /// Matches REST /api/graph/{user_id}/stats endpoint.
    fn graph_stats(&self) -> PyResult<HashMap<String, PyObject>> {
        let stats = self.inner.graph_stats();

        Python::with_gil(|py| {
            let mut dict = HashMap::new();
            dict.insert("node_count".to_string(), stats.node_count.into_py(py));
            dict.insert("edge_count".to_string(), stats.edge_count.into_py(py));
            dict.insert("avg_strength".to_string(), stats.avg_strength.into_py(py));
            dict.insert(
                "potentiated_count".to_string(),
                stats.potentiated_count.into_py(py),
            );
            Ok(dict)
        })
    }

    // === Forget API (Matching REST) ===

    /// Delete a single memory by ID
    ///
    /// Matches REST DELETE /api/memory/{id} endpoint.
    fn forget(&self, memory_id: &str) -> PyResult<bool> {
        // Validate and parse the memory ID as a valid UUID
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        // Use ById to delete the specific memory
        let deleted = self
            .inner
            .forget(ForgetCriteria::ById(MemoryId(uuid)))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete memory: {}", e)))?;

        Ok(deleted > 0)
    }

    /// Delete memories older than specified days
    ///
    /// Matches REST /api/forget/age endpoint.
    fn forget_by_age(&self, days: u32) -> PyResult<usize> {
        self.inner
            .forget(ForgetCriteria::OlderThan(days))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to forget by age: {}", e)))
    }

    /// Delete memories below importance threshold
    ///
    /// Matches REST /api/forget/importance endpoint.
    fn forget_by_importance(&self, threshold: f32) -> PyResult<usize> {
        self.inner
            .forget(ForgetCriteria::LowImportance(threshold))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to forget by importance: {}", e)))
    }

    /// Delete memories matching regex pattern
    ///
    /// Matches REST /api/forget/pattern endpoint.
    fn forget_by_pattern(&self, pattern: &str) -> PyResult<usize> {
        self.inner
            .forget(ForgetCriteria::Pattern(pattern.to_string()))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to forget by pattern: {}", e)))
    }

    /// Delete memories matching any of the specified tags
    ///
    /// Matches REST /api/forget/tags endpoint.
    fn forget_by_tags(&self, tags: Vec<String>) -> PyResult<usize> {
        self.inner
            .forget(ForgetCriteria::ByTags(tags))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to forget by tags: {}", e)))
    }

    /// Delete memories within a date range
    ///
    /// Date strings should be ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    /// Matches REST /api/forget/date endpoint.
    fn forget_by_date(&self, start: &str, end: &str) -> PyResult<usize> {
        let start_dt: DateTime<Utc> = start
            .parse()
            .map_err(|e| PyValueError::new_err(format!("Invalid start date: {}", e)))?;
        let end_dt: DateTime<Utc> = end
            .parse()
            .map_err(|e| PyValueError::new_err(format!("Invalid end date: {}", e)))?;

        self.inner
            .forget(ForgetCriteria::ByDateRange {
                start: start_dt,
                end: end_dt,
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to forget by date: {}", e)))
    }

    /// Delete ALL memories (GDPR compliance - right to erasure)
    ///
    /// Use with caution - this is irreversible.
    fn forget_all(&self) -> PyResult<usize> {
        self.inner
            .forget(ForgetCriteria::All)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to forget all: {}", e)))
    }

    /// Get brain state visualization - shows all memories with activation levels by tier
    ///
    /// Returns 3-tier memory state (working, session, long-term) with activation levels.
    /// Matches REST /api/brain/{user_id} endpoint.
    #[pyo3(signature = (longterm_limit=100))]
    fn brain_state(&self, longterm_limit: usize) -> PyResult<HashMap<String, PyObject>> {
        let working_memories = self.inner.get_working_memories();
        let session_memories = self.inner.get_session_memories();
        let longterm_memories = self
            .inner
            .get_longterm_memories(longterm_limit)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get longterm memories: {}", e))
            })?;

        let working_count = working_memories.len();
        let session_count = session_memories.len();
        let longterm_count = longterm_memories.len();
        let total_count = working_count + session_count + longterm_count;

        Python::with_gil(|py| {
            let mut result = HashMap::new();
            let mut total_activation = 0.0f32;
            let mut total_importance = 0.0f32;

            // Convert working memories (Vec<Arc<Memory>>) to neurons
            let working_neurons: Vec<HashMap<String, PyObject>> = working_memories
                .iter()
                .map(|m| {
                    total_activation += m.activation();
                    total_importance += m.importance();
                    let mut neuron = HashMap::new();
                    neuron.insert("id".to_string(), m.id.0.to_string().into_py(py));
                    neuron.insert(
                        "content_preview".to_string(),
                        m.experience
                            .content
                            .chars()
                            .take(100)
                            .collect::<String>()
                            .into_py(py),
                    );
                    neuron.insert("activation".to_string(), m.activation().into_py(py));
                    neuron.insert("importance".to_string(), m.importance().into_py(py));
                    neuron.insert("tier".to_string(), "working".into_py(py));
                    neuron.insert(
                        "access_count".to_string(),
                        m.metadata_snapshot().access_count.into_py(py),
                    );
                    neuron.insert(
                        "created_at".to_string(),
                        m.created_at.to_rfc3339().into_py(py),
                    );
                    neuron
                })
                .collect();

            // Convert session memories (Vec<Arc<Memory>>) to neurons
            let session_neurons: Vec<HashMap<String, PyObject>> = session_memories
                .iter()
                .map(|m| {
                    total_activation += m.activation();
                    total_importance += m.importance();
                    let mut neuron = HashMap::new();
                    neuron.insert("id".to_string(), m.id.0.to_string().into_py(py));
                    neuron.insert(
                        "content_preview".to_string(),
                        m.experience
                            .content
                            .chars()
                            .take(100)
                            .collect::<String>()
                            .into_py(py),
                    );
                    neuron.insert("activation".to_string(), m.activation().into_py(py));
                    neuron.insert("importance".to_string(), m.importance().into_py(py));
                    neuron.insert("tier".to_string(), "session".into_py(py));
                    neuron.insert(
                        "access_count".to_string(),
                        m.metadata_snapshot().access_count.into_py(py),
                    );
                    neuron.insert(
                        "created_at".to_string(),
                        m.created_at.to_rfc3339().into_py(py),
                    );
                    neuron
                })
                .collect();

            // Convert longterm memories (Vec<Memory>) to neurons
            let longterm_neurons: Vec<HashMap<String, PyObject>> = longterm_memories
                .iter()
                .map(|m| {
                    total_activation += m.activation();
                    total_importance += m.importance();
                    let mut neuron = HashMap::new();
                    neuron.insert("id".to_string(), m.id.0.to_string().into_py(py));
                    neuron.insert(
                        "content_preview".to_string(),
                        m.experience
                            .content
                            .chars()
                            .take(100)
                            .collect::<String>()
                            .into_py(py),
                    );
                    neuron.insert("activation".to_string(), m.activation().into_py(py));
                    neuron.insert("importance".to_string(), m.importance().into_py(py));
                    neuron.insert("tier".to_string(), "longterm".into_py(py));
                    neuron.insert(
                        "access_count".to_string(),
                        m.metadata_snapshot().access_count.into_py(py),
                    );
                    neuron.insert(
                        "created_at".to_string(),
                        m.created_at.to_rfc3339().into_py(py),
                    );
                    neuron
                })
                .collect();

            result.insert("working_memory".to_string(), working_neurons.into_py(py));
            result.insert("session_memory".to_string(), session_neurons.into_py(py));
            result.insert("longterm_memory".to_string(), longterm_neurons.into_py(py));

            // Calculate stats
            let mut stats = HashMap::new();
            stats.insert("total_memories".to_string(), total_count.into_py(py));
            stats.insert("working_count".to_string(), working_count.into_py(py));
            stats.insert("session_count".to_string(), session_count.into_py(py));
            stats.insert("longterm_count".to_string(), longterm_count.into_py(py));
            stats.insert(
                "avg_activation".to_string(),
                if total_count > 0 {
                    (total_activation / total_count as f32).into_py(py)
                } else {
                    0.0f32.into_py(py)
                },
            );
            stats.insert(
                "avg_importance".to_string(),
                if total_count > 0 {
                    (total_importance / total_count as f32).into_py(py)
                } else {
                    0.0f32.into_py(py)
                },
            );
            result.insert("stats".to_string(), stats.into_py(py));

            Ok(result)
        })
    }

    /// Get a report of memory consolidation activity
    ///
    /// Shows memory strengthening/decay events, edge formation, fact extraction,
    /// and maintenance cycles. Use this to understand how memories are evolving.
    /// Matches REST /api/consolidation/report endpoint.
    ///
    /// Args:
    ///     since: Start of report period (ISO 8601 format). Defaults to 24 hours ago.
    ///     until: End of report period (ISO 8601 format). Defaults to now.
    #[pyo3(signature = (since=None, until=None))]
    fn consolidation_report(
        &self,
        since: Option<&str>,
        until: Option<&str>,
    ) -> PyResult<HashMap<String, PyObject>> {
        // Parse since timestamp (default: 24 hours ago)
        let since_dt: DateTime<Utc> = if let Some(s) = since {
            s.parse()
                .map_err(|e| PyValueError::new_err(format!("Invalid 'since' timestamp: {}", e)))?
        } else {
            Utc::now() - chrono::Duration::hours(24)
        };

        // Parse until timestamp (default: now)
        let until_dt: Option<DateTime<Utc>> =
            if let Some(u) = until {
                Some(u.parse().map_err(|e| {
                    PyValueError::new_err(format!("Invalid 'until' timestamp: {}", e))
                })?)
            } else {
                None
            };

        let report = self.inner.get_consolidation_report(since_dt, until_dt);

        Python::with_gil(|py| {
            let mut result = HashMap::new();

            // Period
            let mut period = HashMap::new();
            period.insert(
                "start".to_string(),
                report.period.start.to_rfc3339().into_py(py),
            );
            period.insert(
                "end".to_string(),
                report.period.end.to_rfc3339().into_py(py),
            );
            result.insert("period".to_string(), period.into_py(py));

            // Statistics (from report.statistics)
            let mut stats = HashMap::new();
            stats.insert(
                "total_memories".to_string(),
                report.statistics.total_memories.into_py(py),
            );
            stats.insert(
                "memories_strengthened".to_string(),
                report.statistics.memories_strengthened.into_py(py),
            );
            stats.insert(
                "memories_decayed".to_string(),
                report.statistics.memories_decayed.into_py(py),
            );
            stats.insert(
                "memories_at_risk".to_string(),
                report.statistics.memories_at_risk.into_py(py),
            );
            stats.insert(
                "edges_formed".to_string(),
                report.statistics.edges_formed.into_py(py),
            );
            stats.insert(
                "edges_strengthened".to_string(),
                report.statistics.edges_strengthened.into_py(py),
            );
            stats.insert(
                "edges_potentiated".to_string(),
                report.statistics.edges_potentiated.into_py(py),
            );
            stats.insert(
                "edges_pruned".to_string(),
                report.statistics.edges_pruned.into_py(py),
            );
            stats.insert(
                "facts_extracted".to_string(),
                report.statistics.facts_extracted.into_py(py),
            );
            stats.insert(
                "facts_reinforced".to_string(),
                report.statistics.facts_reinforced.into_py(py),
            );
            stats.insert(
                "maintenance_cycles".to_string(),
                report.statistics.maintenance_cycles.into_py(py),
            );
            stats.insert(
                "total_maintenance_duration_ms".to_string(),
                report.statistics.total_maintenance_duration_ms.into_py(py),
            );
            result.insert("stats".to_string(), stats.into_py(py));

            // Event count (sum of all event lists)
            let event_count = report.strengthened_memories.len()
                + report.decayed_memories.len()
                + report.formed_associations.len()
                + report.strengthened_associations.len()
                + report.potentiated_associations.len()
                + report.pruned_associations.len()
                + report.extracted_facts.len()
                + report.reinforced_facts.len();
            result.insert("event_count".to_string(), event_count.into_py(py));

            // Strengthened memories
            let strengthened: Vec<HashMap<String, PyObject>> = report
                .strengthened_memories
                .iter()
                .map(|m| {
                    let mut mem = HashMap::new();
                    mem.insert("memory_id".to_string(), m.memory_id.clone().into_py(py));
                    mem.insert(
                        "content_preview".to_string(),
                        m.content_preview.clone().into_py(py),
                    );
                    mem.insert(
                        "activation_before".to_string(),
                        m.activation_before.into_py(py),
                    );
                    mem.insert(
                        "activation_after".to_string(),
                        m.activation_after.into_py(py),
                    );
                    mem.insert("reason".to_string(), m.change_reason.clone().into_py(py));
                    mem.insert(
                        "timestamp".to_string(),
                        m.timestamp.to_rfc3339().into_py(py),
                    );
                    mem
                })
                .collect();
            result.insert(
                "strengthened_memories".to_string(),
                strengthened.into_py(py),
            );

            // Decayed memories
            let decayed: Vec<HashMap<String, PyObject>> = report
                .decayed_memories
                .iter()
                .map(|m| {
                    let mut mem = HashMap::new();
                    mem.insert("memory_id".to_string(), m.memory_id.clone().into_py(py));
                    mem.insert(
                        "content_preview".to_string(),
                        m.content_preview.clone().into_py(py),
                    );
                    mem.insert(
                        "activation_before".to_string(),
                        m.activation_before.into_py(py),
                    );
                    mem.insert(
                        "activation_after".to_string(),
                        m.activation_after.into_py(py),
                    );
                    mem.insert("at_risk".to_string(), m.at_risk.into_py(py));
                    mem.insert(
                        "timestamp".to_string(),
                        m.timestamp.to_rfc3339().into_py(py),
                    );
                    mem
                })
                .collect();
            result.insert("decayed_memories".to_string(), decayed.into_py(py));

            // Formed associations
            let formed: Vec<HashMap<String, PyObject>> = report
                .formed_associations
                .iter()
                .map(|a| {
                    let mut assoc = HashMap::new();
                    assoc.insert(
                        "from_memory_id".to_string(),
                        a.from_memory_id.clone().into_py(py),
                    );
                    assoc.insert(
                        "to_memory_id".to_string(),
                        a.to_memory_id.clone().into_py(py),
                    );
                    assoc.insert("strength".to_string(), a.strength_after.into_py(py));
                    assoc.insert("reason".to_string(), a.reason.clone().into_py(py));
                    assoc.insert(
                        "timestamp".to_string(),
                        a.timestamp.to_rfc3339().into_py(py),
                    );
                    assoc
                })
                .collect();
            result.insert("formed_associations".to_string(), formed.into_py(py));

            // Pruned associations
            let pruned: Vec<HashMap<String, PyObject>> = report
                .pruned_associations
                .iter()
                .map(|a| {
                    let mut assoc = HashMap::new();
                    assoc.insert(
                        "from_memory_id".to_string(),
                        a.from_memory_id.clone().into_py(py),
                    );
                    assoc.insert(
                        "to_memory_id".to_string(),
                        a.to_memory_id.clone().into_py(py),
                    );
                    assoc.insert(
                        "final_strength".to_string(),
                        a.strength_before.unwrap_or(0.0).into_py(py),
                    );
                    assoc.insert("reason".to_string(), a.reason.clone().into_py(py));
                    assoc.insert(
                        "timestamp".to_string(),
                        a.timestamp.to_rfc3339().into_py(py),
                    );
                    assoc
                })
                .collect();
            result.insert("pruned_associations".to_string(), pruned.into_py(py));

            Ok(result)
        })
    }

    /// Get all consolidation events since a given timestamp
    ///
    /// Returns raw consolidation events for detailed analysis.
    /// Matches REST /api/consolidation/events endpoint.
    #[pyo3(signature = (since=None))]
    fn consolidation_events(
        &self,
        since: Option<&str>,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let since_dt: DateTime<Utc> = if let Some(s) = since {
            s.parse()
                .map_err(|e| PyValueError::new_err(format!("Invalid 'since' timestamp: {}", e)))?
        } else {
            Utc::now() - chrono::Duration::hours(24)
        };

        let events = self.inner.get_consolidation_events_since(since_dt);

        Python::with_gil(|py| {
            let result: Vec<HashMap<String, PyObject>> = events
                .iter()
                .map(|event| {
                    let mut evt = HashMap::new();
                    evt.insert("event_type".to_string(), format!("{:?}", event).into_py(py));
                    evt.insert(
                        "timestamp".to_string(),
                        event.timestamp().to_rfc3339().into_py(py),
                    );
                    evt
                })
                .collect();
            Ok(result)
        })
    }

    // === Proactive Context API (Matching npm MCP) ===

    /// Surface relevant memories based on current context
    ///
    /// This is for proactive memory surfacing - finding memories relevant to
    /// the current conversation or task without explicit query.
    ///
    /// Matches REST /api/relevant and npm MCP proactive_context tool.
    ///
    /// Args:
    ///     context: Current conversation context or user message
    ///     semantic_threshold: Minimum similarity score (0.0-1.0, default: 0.45)
    ///     max_results: Maximum memories to return (default: 5)
    ///     memory_types: Filter to specific types (empty = all)
    ///     auto_ingest: Store context as Conversation memory (default: true)
    ///     recency_weight: Weight for recency boost (0.0-1.0, default: 0.2)
    #[pyo3(signature = (
        context,
        semantic_threshold=0.45,
        max_results=5,
        memory_types=None,
        auto_ingest=true,
        recency_weight=0.2
    ))]
    fn proactive_context(
        &mut self,
        context: String,
        semantic_threshold: f32,
        max_results: usize,
        memory_types: Option<Vec<String>>,
        auto_ingest: bool,
        recency_weight: f32,
    ) -> PyResult<HashMap<String, PyObject>> {
        let start = std::time::Instant::now();

        // Auto-ingest: Store context as a Conversation memory
        let ingested_id = if auto_ingest && context.len() > 50 {
            let experience = Experience {
                experience_type: ExperienceType::Conversation,
                content: context.clone(),
                tags: vec!["proactive-context".to_string()],
                ..Default::default()
            };
            match self.inner.remember(experience, None) {
                Ok(id) => Some(id.0.to_string()),
                Err(_) => None,
            }
        } else {
            None
        };

        // Semantic search for relevant memories
        let query = Query {
            user_id: None,
            query_text: Some(context.clone()),
            query_embedding: None,
            time_range: None,
            experience_types: memory_types.map(|types| {
                types
                    .iter()
                    .filter_map(|t| match t.to_lowercase().as_str() {
                        "decision" => Some(ExperienceType::Decision),
                        "learning" => Some(ExperienceType::Learning),
                        "error" => Some(ExperienceType::Error),
                        "context" | "observation" => Some(ExperienceType::Context),
                        "pattern" => Some(ExperienceType::Pattern),
                        "conversation" => Some(ExperienceType::Conversation),
                        "task" => Some(ExperienceType::Task),
                        "discovery" => Some(ExperienceType::Discovery),
                        _ => None,
                    })
                    .collect()
            }),
            importance_threshold: None,
            robot_id: self.robot_id.clone(),
            mission_id: None,
            geo_filter: None,
            action_type: None,
            reward_range: None,
            outcome_type: None,
            failures_only: false,
            anomalies_only: false,
            severity: None,
            tags: None,
            pattern_id: None,
            terrain_type: None,
            confidence_range: None,
            prospective_signals: None,
            episode_id: None,
            max_results: max_results * 2, // Get more for filtering
            retrieval_mode: RetrievalMode::Hybrid,
            offset: 0,
        };

        let memories = self
            .inner
            .recall(&query)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to recall memories: {}", e)))?;

        // Apply semantic threshold and recency weighting
        let now = Utc::now();
        let recency_half_life_hours = 24.0f32;

        let mut scored_memories: Vec<(std::sync::Arc<Memory>, f32, String)> = memories
            .into_iter()
            .filter_map(|m| {
                // Get semantic score from the memory's score if available
                let base_score = m.importance();

                // Apply recency boost
                let age_hours = (now - m.created_at).num_hours() as f32;
                let recency_factor = (-age_hours / recency_half_life_hours).exp();
                let recency_boost = 1.0 + (recency_weight * recency_factor);

                let final_score = base_score * recency_boost;

                if final_score >= semantic_threshold {
                    let reason = if recency_factor > 0.5 {
                        "recent_and_relevant".to_string()
                    } else {
                        "semantic_similarity".to_string()
                    };
                    Some((m, final_score.min(1.0), reason))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score and take top results
        scored_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_memories.truncate(max_results);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Python::with_gil(|py| {
            let mut result = HashMap::new();

            // Surfaced memories
            let memories_list: Vec<HashMap<String, PyObject>> = scored_memories
                .into_iter()
                .map(|(m, score, reason)| {
                    let mut mem = HashMap::new();
                    mem.insert("id".to_string(), m.id.0.to_string().into_py(py));
                    mem.insert(
                        "content".to_string(),
                        m.experience.content.clone().into_py(py),
                    );
                    mem.insert(
                        "memory_type".to_string(),
                        format!("{:?}", m.experience.experience_type).into_py(py),
                    );
                    mem.insert("importance".to_string(), m.importance().into_py(py));
                    mem.insert("relevance_score".to_string(), score.into_py(py));
                    mem.insert("relevance_reason".to_string(), reason.into_py(py));
                    mem.insert(
                        "created_at".to_string(),
                        m.created_at.to_rfc3339().into_py(py),
                    );
                    mem.insert("tags".to_string(), m.experience.tags.clone().into_py(py));
                    mem
                })
                .collect();

            let count = memories_list.len();
            result.insert("memories".to_string(), memories_list.into_py(py));
            result.insert("count".to_string(), count.into_py(py));
            result.insert("latency_ms".to_string(), latency_ms.into_py(py));
            result.insert(
                "ingested_id".to_string(),
                ingested_id
                    .map(|id| id.into_py(py))
                    .unwrap_or_else(|| py.None()),
            );
            result.insert(
                "config".to_string(),
                {
                    let mut cfg = HashMap::new();
                    cfg.insert(
                        "semantic_threshold".to_string(),
                        semantic_threshold.into_py(py),
                    );
                    cfg.insert("max_results".to_string(), max_results.into_py(py));
                    cfg.insert("recency_weight".to_string(), recency_weight.into_py(py));
                    cfg.insert("auto_ingest".to_string(), auto_ingest.into_py(py));
                    cfg
                }
                .into_py(py),
            );

            Ok(result)
        })
    }

    // === Index Health API (Matching npm MCP) ===

    /// Verify vector index integrity
    ///
    /// Diagnoses orphaned memories that are stored but not searchable.
    /// Matches REST /api/index/verify and npm MCP verify_index tool.
    ///
    /// Returns:
    ///     dict with: total_storage, total_indexed, orphaned_count, is_healthy
    fn verify_index(&self) -> PyResult<HashMap<String, PyObject>> {
        let report = self
            .inner
            .verify_index_integrity()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to verify index: {}", e)))?;

        Python::with_gil(|py| {
            let mut result = HashMap::new();
            result.insert(
                "total_storage".to_string(),
                report.total_storage.into_py(py),
            );
            result.insert(
                "total_indexed".to_string(),
                report.total_indexed.into_py(py),
            );
            result.insert(
                "orphaned_count".to_string(),
                report.orphaned_count.into_py(py),
            );
            result.insert("is_healthy".to_string(), report.is_healthy.into_py(py));
            result.insert(
                "orphaned_ids".to_string(),
                report
                    .orphaned_ids
                    .iter()
                    .map(|id| id.0.to_string())
                    .collect::<Vec<_>>()
                    .into_py(py),
            );
            Ok(result)
        })
    }

    /// Repair vector index by re-indexing orphaned memories
    ///
    /// Use when verify_index shows unhealthy status.
    /// Matches REST /api/index/repair and npm MCP repair_index tool.
    ///
    /// Returns:
    ///     dict with: total_storage, total_indexed, repaired, failed, is_healthy
    fn repair_index(&self) -> PyResult<HashMap<String, PyObject>> {
        let (total_storage, total_indexed, repaired, failed) = self
            .inner
            .repair_vector_index()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to repair index: {}", e)))?;

        Python::with_gil(|py| {
            let mut result = HashMap::new();
            result.insert("total_storage".to_string(), total_storage.into_py(py));
            result.insert("total_indexed".to_string(), total_indexed.into_py(py));
            result.insert("repaired".to_string(), repaired.into_py(py));
            result.insert("failed".to_string(), failed.into_py(py));
            result.insert("is_healthy".to_string(), (failed == 0).into_py(py));
            result.insert("success".to_string(), true.into_py(py));
            Ok(result)
        })
    }

    /// Get vector index health metrics
    ///
    /// Returns information about the Vamana index including total vectors,
    /// incremental inserts since last build, and whether rebuild is recommended.
    fn index_health(&self) -> PyResult<HashMap<String, PyObject>> {
        let health = self.inner.index_health();

        Python::with_gil(|py| {
            let mut result = HashMap::new();
            result.insert(
                "total_vectors".to_string(),
                health.total_vectors.into_py(py),
            );
            result.insert(
                "deleted_count".to_string(),
                health.deleted_count.into_py(py),
            );
            result.insert(
                "deletion_ratio".to_string(),
                health.deletion_ratio.into_py(py),
            );
            result.insert(
                "needs_compaction".to_string(),
                health.needs_compaction.into_py(py),
            );
            result.insert(
                "incremental_inserts".to_string(),
                health.incremental_inserts.into_py(py),
            );
            result.insert(
                "needs_rebuild".to_string(),
                health.needs_rebuild.into_py(py),
            );
            result.insert(
                "rebuild_threshold".to_string(),
                health.rebuild_threshold.into_py(py),
            );
            result.insert(
                "healthy".to_string(),
                (!health.needs_rebuild && !health.needs_compaction).into_py(py),
            );
            Ok(result)
        })
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "MemorySystem(robot_id={:?}, mission={:?}, total={})",
            self.robot_id, self.mission_id, stats.total_memories
        )
    }
}

/// Convert Memory to Python dict with all fields
fn memory_to_dict(_py: Python, memory: &Memory) -> PyResult<HashMap<String, PyObject>> {
    Python::with_gil(|py| {
        let mut dict = HashMap::new();

        // Core fields
        dict.insert("id".to_string(), memory.id.0.to_string().into_py(py));
        dict.insert(
            "content".to_string(),
            memory.experience.content.clone().into_py(py),
        );
        dict.insert(
            "experience_type".to_string(),
            format!("{:?}", memory.experience.experience_type).into_py(py),
        );
        dict.insert(
            "entities".to_string(),
            memory.experience.entities.clone().into_py(py),
        );
        dict.insert(
            "metadata".to_string(),
            memory.experience.metadata.clone().into_py(py),
        );
        dict.insert("importance".to_string(), memory.importance().into_py(py));
        dict.insert(
            "access_count".to_string(),
            memory.access_count().into_py(py),
        );
        dict.insert(
            "created_at".to_string(),
            memory.created_at.to_rfc3339().into_py(py),
        );
        dict.insert(
            "last_accessed".to_string(),
            memory.last_accessed().to_rfc3339().into_py(py),
        );
        dict.insert("compressed".to_string(), memory.compressed.into_py(py));

        // Robotics fields
        if let Some(ref robot_id) = memory.experience.robot_id {
            dict.insert("robot_id".to_string(), robot_id.clone().into_py(py));
        }
        if let Some(ref mission_id) = memory.experience.mission_id {
            dict.insert("mission_id".to_string(), mission_id.clone().into_py(py));
        }
        if let Some(ref geo) = memory.experience.geo_location {
            dict.insert("geo_location".to_string(), geo.to_vec().into_py(py));
        }
        if let Some(ref pos) = memory.experience.local_position {
            dict.insert("position".to_string(), pos.to_vec().into_py(py));
        }
        if let Some(heading) = memory.experience.heading {
            dict.insert("heading".to_string(), heading.into_py(py));
        }
        if let Some(ref action_type) = memory.experience.action_type {
            dict.insert("action_type".to_string(), action_type.clone().into_py(py));
        }
        if let Some(reward) = memory.experience.reward {
            dict.insert("reward".to_string(), reward.into_py(py));
        }
        if !memory.experience.sensor_data.is_empty() {
            dict.insert(
                "sensor_data".to_string(),
                memory.experience.sensor_data.clone().into_py(py),
            );
        }

        // Decision fields
        if let Some(ref ctx) = memory.experience.decision_context {
            dict.insert("decision_context".to_string(), ctx.clone().into_py(py));
        }
        if let Some(ref params) = memory.experience.action_params {
            dict.insert("action_params".to_string(), params.clone().into_py(py));
        }
        if let Some(ref outcome_type) = memory.experience.outcome_type {
            dict.insert("outcome_type".to_string(), outcome_type.clone().into_py(py));
        }
        if let Some(ref details) = memory.experience.outcome_details {
            dict.insert("outcome_details".to_string(), details.clone().into_py(py));
        }
        if let Some(confidence) = memory.experience.confidence {
            dict.insert("confidence".to_string(), confidence.into_py(py));
        }
        if !memory.experience.alternatives_considered.is_empty() {
            dict.insert(
                "alternatives_considered".to_string(),
                memory
                    .experience
                    .alternatives_considered
                    .clone()
                    .into_py(py),
            );
        }

        // Environment fields
        if let Some(ref weather) = memory.experience.weather {
            dict.insert("weather".to_string(), weather.clone().into_py(py));
        }
        if let Some(ref terrain) = memory.experience.terrain_type {
            dict.insert("terrain_type".to_string(), terrain.clone().into_py(py));
        }
        if let Some(ref lighting) = memory.experience.lighting {
            dict.insert("lighting".to_string(), lighting.clone().into_py(py));
        }
        if !memory.experience.nearby_agents.is_empty() {
            dict.insert(
                "nearby_agents".to_string(),
                memory.experience.nearby_agents.clone().into_py(py),
            );
        }

        // Failure fields
        dict.insert(
            "is_failure".to_string(),
            memory.experience.is_failure.into_py(py),
        );
        dict.insert(
            "is_anomaly".to_string(),
            memory.experience.is_anomaly.into_py(py),
        );
        if let Some(ref severity) = memory.experience.severity {
            dict.insert("severity".to_string(), severity.clone().into_py(py));
        }
        if let Some(ref recovery) = memory.experience.recovery_action {
            dict.insert("recovery_action".to_string(), recovery.clone().into_py(py));
        }
        if let Some(ref cause) = memory.experience.root_cause {
            dict.insert("root_cause".to_string(), cause.clone().into_py(py));
        }

        // Pattern fields
        if let Some(ref pattern) = memory.experience.pattern_id {
            dict.insert("pattern_id".to_string(), pattern.clone().into_py(py));
        }
        if let Some(ref predicted) = memory.experience.predicted_outcome {
            dict.insert(
                "predicted_outcome".to_string(),
                predicted.clone().into_py(py),
            );
        }
        if let Some(accurate) = memory.experience.prediction_accurate {
            dict.insert("prediction_accurate".to_string(), accurate.into_py(py));
        }
        if !memory.experience.tags.is_empty() {
            dict.insert(
                "tags".to_string(),
                memory.experience.tags.clone().into_py(py),
            );
        }

        Ok(dict)
    })
}

/// Python module definition
#[pymodule]
fn shodh_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<PyMemorySystem>()?;

    // Robotics types
    m.add_class::<PyPosition>()?;
    m.add_class::<PyGeoLocation>()?;
    m.add_class::<PyGeoFilter>()?;

    // Decision & Learning types
    m.add_class::<PyDecisionContext>()?;
    m.add_class::<PyOutcome>()?;
    m.add_class::<PyEnvironment>()?;

    // Version and metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Shodh-Memory: AI Memory System for Autonomous Robots & Drones\n\n\
                       Features:\n\
                       - Position(x, y, z) for local coordinates\n\
                       - GeoLocation(lat, lon, alt) for GPS\n\
                       - DecisionContext for action-outcome learning\n\
                       - Outcome for decision results\n\
                       - Environment for weather, terrain, lighting\n\
                       - Failure tracking and anomaly detection\n\
                       - Pattern learning and predictions\n\
                       - 100% offline capable",
    )?;

    Ok(())
}
