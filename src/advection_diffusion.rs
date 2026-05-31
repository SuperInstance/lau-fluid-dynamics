//! Advection-diffusion equation solver with stabilization.
//!
//! Solves ∂φ/∂t + u·∇φ = D·∇²φ
//! using operator splitting: advection (upwind) + diffusion (central).

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Advection-diffusion solver on a 2D uniform grid.
#[derive(Clone, Serialize, Deserialize)]
pub struct AdvectionDiffusionSolver {
    /// Scalar field φ [ny * nx]
    pub phi: DVector<f64>,
    /// Velocity field u (x-component) [ny * nx]
    pub velocity_x: DVector<f64>,
    /// Velocity field v (y-component) [ny * nx]
    pub velocity_y: DVector<f64>,
    /// Diffusion coefficient
    pub diffusivity: f64,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub length_x: f64,
    pub length_y: f64,
    pub time: f64,
}

impl AdvectionDiffusionSolver {
    /// Create a new solver.
    pub fn new(nx: usize, ny: usize, diffusivity: f64, length_x: f64, length_y: f64) -> Self {
        let n = nx * ny;
        Self {
            phi: DVector::zeros(n),
            velocity_x: DVector::zeros(n),
            velocity_y: DVector::zeros(n),
            diffusivity,
            nx,
            ny,
            dx: length_x / (nx - 1).max(1) as f64,
            dy: length_y / (ny - 1).max(1) as f64,
            length_x,
            length_y,
            time: 0.0,
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Set uniform velocity field.
    pub fn set_uniform_velocity(&mut self, u: f64, v: f64) {
        for k in 0..self.velocity_x.len() {
            self.velocity_x[k] = u;
            self.velocity_y[k] = v;
        }
    }

    /// Initialize φ with a Gaussian blob centered at (cx, cy).
    pub fn init_gaussian(&mut self, cx: f64, cy: f64, sigma: f64, amplitude: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = i as f64 * self.dx;
                let y = j as f64 * self.dy;
                let r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                let idx = self.idx(i, j);
                self.phi[idx] = amplitude * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    /// Compute Peclet number (ratio of advection to diffusion).
    pub fn peclet_number(&self, characteristic_length: f64) -> f64 {
        let u_max = self.velocity_x.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let v_max = self.velocity_y.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let vel_max = u_max.max(v_max);
        vel_max * characteristic_length / self.diffusivity.max(1e-20)
    }

    /// Compute stable time step.
    pub fn stable_dt(&self, cfl: f64) -> f64 {
        let u_max = self.velocity_x.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let v_max = self.velocity_y.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let vel_max = u_max.max(v_max).max(1e-10);
        let dt_adv = cfl * self.dx.min(self.dy) / vel_max;
        let dt_diff = 0.5 * (self.dx * self.dx * self.dy * self.dy)
            / (self.diffusivity.max(1e-20) * (self.dx * self.dx + self.dy * self.dy));
        dt_adv.min(dt_diff)
    }

    /// Advance one step with upwind advection + central diffusion + SUPG stabilization.
    pub fn step(&mut self, dt: f64) {
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dy = self.dy;
        let d = self.diffusivity;
        let mut new_phi = self.phi.clone();

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = self.idx(i, j);
                let phi_ij = self.phi[k];
                let u = self.velocity_x[k];
                let v = self.velocity_y[k];

                // Upwind advection
                let dphidx = if u >= 0.0 {
                    (phi_ij - self.phi[self.idx(i - 1, j)]) / dx
                } else {
                    (self.phi[self.idx(i + 1, j)] - phi_ij) / dx
                };
                let dphidy = if v >= 0.0 {
                    (phi_ij - self.phi[self.idx(i, j - 1)]) / dy
                } else {
                    (self.phi[self.idx(i, j + 1)] - phi_ij) / dy
                };

                // Central diffusion
                let laplacian = (self.phi[self.idx(i + 1, j)] - 2.0 * phi_ij + self.phi[self.idx(i - 1, j)])
                    / (dx * dx)
                    + (self.phi[self.idx(i, j + 1)] - 2.0 * phi_ij + self.phi[self.idx(i, j - 1)])
                    / (dy * dy);

                // SUPG stabilization parameter
                let vel_mag = (u * u + v * v).sqrt().max(1e-10);
                let h = dx.min(dy);
                let pe_h = vel_mag * h / (2.0 * d.max(1e-20));
                let tau_supg = h / (2.0 * vel_mag) * (1.0 / (pe_h + 1e-10)).min(1.0).coth() - 1.0 / (pe_h + 1e-10).max(1e-10);

                // Residual-based stabilization
                let residual = -(u * dphidx + v * dphidy) + d * laplacian;
                let supg_stab = tau_supg * vel_mag * residual;

                new_phi[k] = phi_ij + dt * (-u * dphidx - v * dphidy + d * laplacian + supg_stab);
            }
        }

        // Zero-flux boundary conditions
        for i in 0..nx {
            new_phi[self.idx(i, 0)] = new_phi[self.idx(i, 1)];
            new_phi[self.idx(i, ny - 1)] = new_phi[self.idx(i, ny - 2)];
        }
        for j in 0..ny {
            new_phi[self.idx(0, j)] = new_phi[self.idx(1, j)];
            new_phi[self.idx(nx - 1, j)] = new_phi[self.idx(nx - 2, j)];
        }

        self.phi = new_phi;
        self.time += dt;
    }

    /// Advance multiple steps.
    pub fn advance(&mut self, dt: f64, steps: usize) {
        for _ in 0..steps {
            self.step(dt);
        }
    }

    /// Total mass (integral of φ).
    pub fn total_mass(&self) -> f64 {
        self.phi.iter().map(|&p| p * self.dx * self.dy).sum()
    }

    /// Maximum value of φ.
    pub fn max_phi(&self) -> f64 {
        self.phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum value of φ.
    pub fn min_phi(&self) -> f64 {
        self.phi.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}

/// Helper trait for coth (absent from std).
trait FloatExt {
    fn coth(self) -> f64;
}

impl FloatExt for f64 {
    fn coth(self) -> f64 {
        let x = self.abs();
        if x < 1e-4 {
            1.0 / self + self / 3.0
        } else {
            (2.0 * x).exp() + 1.0 / ((2.0 * x).exp() - 1.0)
        }
    }
}
