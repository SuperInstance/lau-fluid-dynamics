//! Agent flow dynamics — modeling agent populations as fluids.
//!
//! Applies fluid dynamics principles to predict macro-behavior of
//! agent populations (e.g., users, bots, particles in a system).

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::flow_regime::FlowRegime;
use crate::navier_stokes::NavierStokes2D;
use crate::lattice_boltzmann::LatticeBoltzmannD2Q9;
use crate::advection_diffusion::AdvectionDiffusionSolver;

/// Parameters for an agent flow model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentFlowParams {
    /// Number of agents (affects density)
    pub agent_count: f64,
    /// Characteristic agent speed
    pub agent_speed: f64,
    /// Domain size
    pub domain_size: f64,
    /// Agent "viscosity" — resistance to flow / mixing rate
    pub interaction_strength: f64,
}

/// Agent flow dynamics model.
///
/// Treats a population of agents as a continuous fluid where:
/// - Density = agent concentration
/// - Velocity = average agent movement direction/speed
/// - Viscosity = interaction strength (cohesion/repulsion)
/// - Reynolds number classifies collective behavior regime
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentFlowModel {
    /// Model parameters
    pub params: AgentFlowParams,
    /// 2D Navier-Stokes solver for agent density/velocity
    pub flow: NavierStokes2D,
    /// Scalar field representing agent property (e.g., activation level)
    pub property: AdvectionDiffusionSolver,
    /// LBM solver for mesoscopic agent dynamics
    pub lbm: LatticeBoltzmannD2Q9,
    /// Reynolds number for the agent flow
    pub reynolds_number: f64,
}

impl AgentFlowModel {
    /// Create a new agent flow model.
    pub fn new(grid_size: usize, params: AgentFlowParams) -> Self {
        let viscosity = params.interaction_strength;
        let domain = params.domain_size;

        let flow = NavierStokes2D::new(grid_size, grid_size, viscosity, domain, domain);
        let property = AdvectionDiffusionSolver::new(
            grid_size,
            grid_size,
            viscosity * 0.5,
            domain,
            domain,
        );
        let lbm_tau = 3.0 * viscosity + 0.5;
        let lbm = LatticeBoltzmannD2Q9::new(grid_size, grid_size, lbm_tau);

        let reynolds_number = params.agent_speed * params.domain_size / viscosity;

        Self {
            params,
            flow,
            property,
            lbm,
            reynolds_number,
        }
    }

    /// Classify the agent population behavior regime.
    pub fn behavior_regime(&self) -> FlowRegime {
        use crate::flow_regime::ReynoldsNumber;
        let re = ReynoldsNumber::new(
            self.params.agent_speed,
            self.params.domain_size,
            self.params.interaction_strength,
        );
        re.regime()
    }

    /// Initialize with uniform agent distribution.
    pub fn init_uniform(&mut self, density: f64, ux: f64, uy: f64) {
        self.flow.init_uniform(ux, uy);
        self.property.set_uniform_velocity(ux, uy);
        self.property.phi.fill(density);
        self.lbm.init_equilibrium(density, ux, uy);
    }

    /// Initialize with agent clustering (high density at center).
    pub fn init_clustered(&mut self, cx: f64, cy: f64, sigma: f64) {
        let nx = self.flow.nx;
        let ny = self.flow.ny;
        let dx = self.flow.dx;
        let dy = self.flow.dy;

        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                let k = j * nx + i;
                let r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                let density = self.params.agent_count * (-r2 / (2.0 * sigma * sigma)).exp();
                self.property.phi[k] = density;
            }
        }
    }

    /// Compute the total agent mass in the system.
    pub fn total_agent_mass(&self) -> f64 {
        self.property.total_mass()
    }

    /// Compute the average agent density.
    pub fn average_density(&self) -> f64 {
        self.property.phi.iter().cloned().sum::<f64>() / self.property.phi.len() as f64
    }

    /// Compute the density variance (measure of clustering).
    pub fn density_variance(&self) -> f64 {
        let mean = self.average_density();
        let var: f64 = self
            .property
            .phi
            .iter()
            .map(|&d| (d - mean) * (d - mean))
            .sum::<f64>()
            / self.property.phi.len() as f64;
        var
    }

    /// Detect clustering: return regions where density exceeds threshold.
    pub fn detect_clusters(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let mut clusters = Vec::new();
        let nx = self.flow.nx;
        let ny = self.flow.ny;

        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if self.property.phi[k] > threshold {
                    clusters.push((i, j, self.property.phi[k]));
                }
            }
        }
        clusters
    }

    /// Compute the flux of agents through a vertical line at column i.
    pub fn flux_through_column(&self, i: usize) -> f64 {
        let nx = self.flow.nx;
        let mut flux = 0.0;
        for j in 0..self.flow.ny {
            let k = j * nx + i;
            flux += self.property.phi[k] * self.flow.u[k] * self.flow.dy;
        }
        flux
    }

    /// Advance the model by one time step.
    pub fn step(&mut self, dt: f64, pressure_iters: usize) {
        // Update velocity field
        self.flow.step(dt, pressure_iters);

        // Synchronize property advection with updated velocity
        for k in 0..self.flow.u.len() {
            self.property.velocity_x[k] = self.flow.u[k];
            self.property.velocity_y[k] = self.flow.v[k];
        }

        // Advance property field
        self.property.step(dt);

        // Advance LBM
        self.lbm.step();
    }

    /// Advance multiple steps.
    pub fn advance(&mut self, dt: f64, steps: usize, pressure_iters: usize) {
        for _ in 0..steps {
            self.step(dt, pressure_iters);
        }
    }

    /// Get a summary of the current state.
    pub fn summary(&self) -> AgentFlowSummary {
        AgentFlowSummary {
            reynolds_number: self.reynolds_number,
            regime: self.behavior_regime().to_string(),
            total_mass: self.total_agent_mass(),
            average_density: self.average_density(),
            density_variance: self.density_variance(),
            kinetic_energy: self.flow.kinetic_energy(),
        }
    }
}

/// Summary of agent flow state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentFlowSummary {
    pub reynolds_number: f64,
    pub regime: String,
    pub total_mass: f64,
    pub average_density: f64,
    pub density_variance: f64,
    pub kinetic_energy: f64,
}
