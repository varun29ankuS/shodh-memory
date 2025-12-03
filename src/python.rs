// Python bindings for shodh-memory using PyO3
//
// Comprehensive robotics memory system with:
// - Position/GeoLocation tracking
// - Decision tree learning (action → outcome → reward)
// - Sensor data patterns and anomaly detection
// - Environmental context (weather, terrain, lighting)
// - Failure tracking and recovery patterns
// - Learned behaviors and predictions

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::path::PathBuf;
use std::collections::HashMap;

use crate::memory::{MemorySystem, MemoryConfig, Query, RetrievalMode};
use crate::memory::types::{Experience, ExperienceType, Memory, GeoFilter};

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
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
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
        PyGeoLocation { latitude, longitude, altitude }
    }

    fn to_list(&self) -> Vec<f64> {
        vec![self.latitude, self.longitude, self.altitude]
    }

    fn __repr__(&self) -> String {
        format!("GeoLocation(lat={}, lon={}, alt={})", self.latitude, self.longitude, self.altitude)
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
        PyGeoFilter { latitude, longitude, radius_meters }
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
        format!("GeoFilter(lat={}, lon={}, radius={}m)", self.latitude, self.longitude, self.radius_meters)
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
        format!("DecisionContext(state={:?}, confidence={:?})", self.state, self.confidence)
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
        PyOutcome { outcome_type, details, reward, prediction_accurate }
    }

    fn is_success(&self) -> bool {
        self.outcome_type == "success"
    }

    fn is_failure(&self) -> bool {
        self.outcome_type == "failure"
    }

    fn __repr__(&self) -> String {
        format!("Outcome(type={}, reward={:?})", self.outcome_type, self.reward)
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
        format!("Environment(terrain={:?}, lighting={:?})", self.terrain_type, self.lighting)
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

        let inner = MemorySystem::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create memory system: {}", e)))?;

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

    // === Core Recording (Full API) ===

    /// Record an experience with full robotics and decision-making support
    #[pyo3(signature = (
        content,
        experience_type="observation",
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
    fn record(
        &mut self,
        content: String,
        experience_type: &str,
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
        let exp_type = match experience_type.to_lowercase().as_str() {
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
            outcomes: outcome.as_ref().and_then(|o| o.details.clone()).map(|d| vec![d]).unwrap_or_default(),
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
            alternatives_considered: decision_context.map(|d| d.alternatives.clone()).unwrap_or_default(),
            // Environment fields
            weather: environment.map(|e| e.weather.clone()),
            terrain_type: environment.and_then(|e| e.terrain_type.clone()),
            lighting: environment.and_then(|e| e.lighting.clone()),
            nearby_agents: environment.map(|e| e.nearby_agents.clone()).unwrap_or_default(),
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
        };

        let memory_id = self.inner.record(experience)
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
        self.record(
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
        self.record(
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
        self.record(
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

        self.record(
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

        self.record(
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
        self.record(
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

    // === Query Methods ===

    /// Retrieve memories with comprehensive filtering
    #[pyo3(signature = (
        query,
        max_results=10,
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
    fn retrieve(
        &self,
        query: String,
        max_results: usize,
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
            "spatial" => RetrievalMode::Spatial,
            "mission" => RetrievalMode::Mission,
            "action_outcome" => RetrievalMode::ActionOutcome,
            _ => return Err(PyValueError::new_err(format!("Unknown retrieval mode: {}", mode))),
        };

        let confidence_range = match (min_confidence, max_confidence) {
            (Some(min), Some(max)) => Some((min, max)),
            (Some(min), None) => Some((min, 1.0)),
            (None, Some(max)) => Some((0.0, max)),
            (None, None) => None,
        };

        let query_obj = Query {
            query_text: Some(query),
            query_embedding: None,
            time_range: None,
            experience_types: None,
            importance_threshold: min_importance,
            robot_id: self.robot_id.clone(),
            mission_id,
            geo_filter: geo_filter.map(|f| GeoFilter::new(f.latitude, f.longitude, f.radius_meters)),
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
            max_results,
            retrieval_mode,
        };

        let memories = self.inner.retrieve(&query_obj)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to retrieve memories: {}", e)))?;

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

        self.retrieve(
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
        self.retrieve(
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
        self.retrieve(
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
        self.retrieve(
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
        result.insert("working_memory_count".to_string(), stats.working_memory_count);
        result.insert("session_memory_count".to_string(), stats.session_memory_count);
        result.insert("long_term_memory_count".to_string(), stats.long_term_memory_count);
        result.insert("compressed_count".to_string(), stats.compressed_count);
        result.insert("promotions_to_session".to_string(), stats.promotions_to_session);
        result.insert("promotions_to_longterm".to_string(), stats.promotions_to_longterm);
        result.insert("total_retrievals".to_string(), stats.total_retrievals);

        Ok(result)
    }

    fn flush(&self) -> PyResult<()> {
        self.inner.flush_storage()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to flush: {}", e)))
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
        dict.insert("content".to_string(), memory.experience.content.clone().into_py(py));
        dict.insert("experience_type".to_string(), format!("{:?}", memory.experience.experience_type).into_py(py));
        dict.insert("entities".to_string(), memory.experience.entities.clone().into_py(py));
        dict.insert("metadata".to_string(), memory.experience.metadata.clone().into_py(py));
        dict.insert("importance".to_string(), memory.importance().into_py(py));
        dict.insert("access_count".to_string(), memory.access_count().into_py(py));
        dict.insert("created_at".to_string(), memory.created_at.to_rfc3339().into_py(py));
        dict.insert("last_accessed".to_string(), memory.last_accessed().to_rfc3339().into_py(py));
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
            dict.insert("sensor_data".to_string(), memory.experience.sensor_data.clone().into_py(py));
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
            dict.insert("alternatives_considered".to_string(), memory.experience.alternatives_considered.clone().into_py(py));
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
            dict.insert("nearby_agents".to_string(), memory.experience.nearby_agents.clone().into_py(py));
        }

        // Failure fields
        dict.insert("is_failure".to_string(), memory.experience.is_failure.into_py(py));
        dict.insert("is_anomaly".to_string(), memory.experience.is_anomaly.into_py(py));
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
            dict.insert("predicted_outcome".to_string(), predicted.clone().into_py(py));
        }
        if let Some(accurate) = memory.experience.prediction_accurate {
            dict.insert("prediction_accurate".to_string(), accurate.into_py(py));
        }
        if !memory.experience.tags.is_empty() {
            dict.insert("tags".to_string(), memory.experience.tags.clone().into_py(py));
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
    m.add("__doc__", "Shodh-Memory: AI Memory System for Autonomous Robots & Drones\n\n\
                       Features:\n\
                       - Position(x, y, z) for local coordinates\n\
                       - GeoLocation(lat, lon, alt) for GPS\n\
                       - DecisionContext for action-outcome learning\n\
                       - Outcome for decision results\n\
                       - Environment for weather, terrain, lighting\n\
                       - Failure tracking and anomaly detection\n\
                       - Pattern learning and predictions\n\
                       - 100% offline capable")?;

    Ok(())
}
