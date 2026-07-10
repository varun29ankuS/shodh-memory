//! Native spaCy-compatible inference runtime.
//!
//! v1 scope: tokenizer + tok2vec + tagger + attribute_ruler + NER, loading
//! weights/config exported from a spaCy pipeline (see ../../export). Compiles
//! native (for tests) and to wasm32 (browser; added in a later story).

// Vendored third-party crate (see NOTICE): tracked against upstream, not linted
// to this workspace's clippy style. Behavior is pinned by golden-parity tests,
// not by lints, so we allow upstream idioms (e.g. single-pass `for` loops that
// destructure-and-return) rather than diverge from the source we vendor.
#![allow(clippy::all)]

pub mod attribute_ruler;
pub mod features;
pub mod hash;
pub mod lemmatizer;
pub mod lexeme;
pub mod matcher;
pub mod ml;
pub mod model;
pub mod ner;
pub mod parser;
pub mod pipeline;
pub mod tagger;
pub mod tok2vec;
pub mod tokenizer;
pub mod transition;
pub mod vectors;

#[cfg(all(target_arch = "wasm32", feature = "bindgen"))]
mod wasm;

#[cfg(all(target_arch = "wasm32", feature = "capi"))]
mod capi;
