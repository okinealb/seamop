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
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register(module)
}
