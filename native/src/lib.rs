mod compact;
mod energy;
mod engine;
mod seam;

pub(crate) const CHANNELS: usize = 3;

pub use engine::{plan_gradient, EngineError, GradientPlan};
