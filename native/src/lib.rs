mod compact;
mod energy;
mod engine;
mod forward;
mod image;
mod seam;
mod transpose;

pub(crate) const CHANNELS: usize = 3;

pub use engine::{plan_forward, plan_gradient, EngineError, GradientPlan};
