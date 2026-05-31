//! # lau-fluid-dynamics
//!
//! Computational fluid dynamics in Rust: Navier-Stokes, Euler equations,
//! Lattice Boltzmann (D2Q9), vortex methods, advection-diffusion, and
//! agent flow dynamics for macro-behavior prediction.

pub mod navier_stokes;
pub mod euler;
pub mod lattice_boltzmann;
pub mod advection_diffusion;
pub mod vortex;
pub mod potential;
pub mod flow_regime;
pub mod cavity;
pub mod agent_flow;

pub use navier_stokes::{NavierStokes1D, NavierStokes2D};
pub use euler::EulerSolver;
pub use lattice_boltzmann::LatticeBoltzmannD2Q9;
pub use advection_diffusion::AdvectionDiffusionSolver;
pub use vortex::{PointVortex, VortexSheet, VortexSystem};
pub use potential::{StreamFunction, VelocityPotential};
pub use flow_regime::{ReynoldsNumber, FlowRegime};
pub use cavity::LidDrivenCavity;
pub use agent_flow::AgentFlowModel;
