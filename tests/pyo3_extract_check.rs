//! Empirical check that the six `Clone` pyclasses still extract by value under
//! pyo3 0.29.
//!
//! pyo3 0.28 deprecated the blanket `FromPyObject` impl for `#[pyclass]` types
//! that implement `Clone`, and 0.30 flips it off by default. The bindings opt
//! back in with `#[pyclass(..., from_py_object)]`. This test proves the opt-in
//! actually produces a working `FromPyObject` for each of the six, rather than
//! trusting that the attribute is spelled right.

#![cfg(feature = "python")]

use pyo3::prelude::*;
use shodh_memory::python::{
    PyDecisionContext, PyEnvironment, PyGeoFilter, PyGeoLocation, PyOutcome, PyPosition,
};
use std::collections::HashMap;

/// Round-trip one pyclass value: Rust -> Python object -> Rust, by value.
/// The `extract()` is the operation that needs `FromPyObject`.
fn roundtrip<T>(py: Python<'_>, value: T) -> T
where
    T: Clone + pyo3::PyClass<Frozen = pyo3::pyclass::boolean_struct::False> + for<'a> FromPyObject<'a>,
{
    let obj = Py::new(py, value).expect("construct pyclass instance");
    obj.bind(py)
        .as_any()
        .extract::<T>()
        .expect("extract pyclass by value")
}

#[test]
fn clone_pyclasses_extract_by_value() {
    Python::initialize();
    Python::attach(|py| {
        let position = roundtrip(
            py,
            PyPosition {
                x: 1.5,
                y: -2.5,
                z: 3.25,
            },
        );
        assert_eq!((position.x, position.y, position.z), (1.5, -2.5, 3.25));

        let geo = roundtrip(
            py,
            PyGeoLocation {
                latitude: 39.2904,
                longitude: -76.6122,
                altitude: 10.0,
            },
        );
        assert_eq!(
            (geo.latitude, geo.longitude, geo.altitude),
            (39.2904, -76.6122, 10.0)
        );

        let filter = roundtrip(
            py,
            PyGeoFilter {
                latitude: 39.2904,
                longitude: -76.6122,
                radius_meters: 500.0,
            },
        );
        assert_eq!(
            (filter.latitude, filter.longitude, filter.radius_meters),
            (39.2904, -76.6122, 500.0)
        );

        let ctx = roundtrip(
            py,
            PyDecisionContext {
                state: HashMap::from([("battery_low".to_string(), "true".to_string())]),
                action_params: HashMap::from([("speed".to_string(), "0.5".to_string())]),
                confidence: Some(0.75),
                alternatives: vec!["hold".to_string()],
            },
        );
        assert_eq!(ctx.state.get("battery_low").map(String::as_str), Some("true"));
        assert_eq!(
            ctx.action_params.get("speed").map(String::as_str),
            Some("0.5")
        );
        assert_eq!(ctx.confidence, Some(0.75));
        assert_eq!(ctx.alternatives, vec!["hold".to_string()]);

        let outcome = roundtrip(
            py,
            PyOutcome {
                outcome_type: "success".to_string(),
                details: Some("landed".to_string()),
                reward: Some(0.9),
                prediction_accurate: Some(true),
            },
        );
        assert_eq!(outcome.outcome_type, "success");
        assert_eq!(outcome.details.as_deref(), Some("landed"));
        assert_eq!(outcome.reward, Some(0.9));
        assert_eq!(outcome.prediction_accurate, Some(true));

        let env = roundtrip(
            py,
            PyEnvironment {
                weather: HashMap::from([("visibility".to_string(), "good".to_string())]),
                terrain_type: Some("urban".to_string()),
                lighting: Some("dim".to_string()),
                nearby_agents: vec![HashMap::from([(
                    "id".to_string(),
                    "drone_002".to_string(),
                )])],
            },
        );
        assert_eq!(
            env.weather.get("visibility").map(String::as_str),
            Some("good")
        );
        assert_eq!(env.terrain_type.as_deref(), Some("urban"));
        assert_eq!(env.lighting.as_deref(), Some("dim"));
        assert_eq!(env.nearby_agents.len(), 1);
    });
}
