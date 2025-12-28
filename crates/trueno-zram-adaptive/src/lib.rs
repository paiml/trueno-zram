//! ML-driven compression algorithm selection for trueno-zram.
//!
//! This crate provides entropy-based analysis and machine learning models
//! to select the optimal compression algorithm for each memory page.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]

pub mod classifier;
pub mod entropy;
pub mod model;

pub use classifier::PageClassifier;
pub use entropy::EntropyCalculator;
