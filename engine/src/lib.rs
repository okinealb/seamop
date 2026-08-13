//! Rust engine behind the private `seamop._engine` Python module.
//!
//! The engine receives one contiguous RGB `u8` image, performs a complete
//! built-in resize plan, and returns the carved image with a source-sized
//! removal mask. Python owns input normalization, public validation, and
//! custom energy callables.

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod bindings;
mod compact;
mod energy;
mod engine;
mod forward;
mod image;
mod seam;
mod transpose;

pub(crate) const CHANNELS: usize = 3;

pub use engine::{plan, plan_forward, EngineError, Plan};

#[pymodule]
fn _engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register(module)
}
