//! PyO3 bindings for the private `seamop._engine` module.
//!
//! This boundary checks the channel count and memory layout, copies the input
//! into Rust-owned storage, releases the GIL during planning, and converts the
//! result and removal mask back into NumPy arrays. Engine errors become Python
//! exceptions here rather than leaking Rust details into the package API.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::engine::{self, EngineError, Plan};
use crate::CHANNELS;

type PythonPlan = (Py<PyArray3<u8>>, Py<PyArray2<bool>>);
type Planner = fn(
    image: &[u8],
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Plan, EngineError>;

#[pyfunction]
fn plan(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, u8>,
    target_height: usize,
    target_width: usize,
) -> PyResult<PythonPlan> {
    run_plan(py, image, target_height, target_width, engine::plan)
}

#[pyfunction]
fn plan_forward(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, u8>,
    target_height: usize,
    target_width: usize,
) -> PyResult<PythonPlan> {
    run_plan(py, image, target_height, target_width, engine::plan_forward)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(plan, module)?)?;
    module.add_function(wrap_pyfunction!(plan_forward, module)?)?;
    Ok(())
}

fn run_plan(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, u8>,
    target_height: usize,
    target_width: usize,
    planner: Planner,
) -> PyResult<PythonPlan> {
    let image_array = image.as_array();
    let shape = image_array.shape();
    let height = shape[0];
    let width = shape[1];
    if shape[2] != CHANNELS {
        return Err(PyValueError::new_err(format!(
            "image must have exactly {CHANNELS} RGB channels; got {}",
            shape[2]
        )));
    }

    let image = image.as_slice().map_err(|_| {
        PyValueError::new_err("image must be C-contiguous for engine planning")
    })?;
    let image = image.to_vec();

    let plan = py
        .detach(|| planner(&image, height, width, target_height, target_width))
        .map_err(map_engine_error)?;

    let result = Array3::from_shape_vec(
        (plan.target_height, plan.target_width, CHANNELS),
        plan.result,
    )
    .map_err(|_| PyValueError::new_err("engine result has an invalid shape"))?
    .into_pyarray(py)
    .unbind();
    let removed = Array2::from_shape_vec(
        (plan.source_height, plan.source_width),
        plan.removed_mask,
    )
    .map_err(|_| {
        PyValueError::new_err("engine removal mask has an invalid shape")
    })?
    .into_pyarray(py)
    .unbind();

    Ok((result, removed))
}

fn map_engine_error(error: EngineError) -> PyErr {
    let message = error.to_string();
    match error {
        EngineError::NoSeam => PyRuntimeError::new_err(message),
        EngineError::SizeOverflow => PyOverflowError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}
