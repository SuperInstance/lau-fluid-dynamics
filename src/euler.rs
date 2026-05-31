//! Euler equations for inviscid (zero viscosity) flow.
//!
//! Solves the 2D Euler equations using a Lax-Friedrichs scheme:
//! ∂ρ/∂t + ∇·(ρu) = 0  (mass conservation)
//! ∂(ρu)/∂t + ∇·(ρu⊗u + pI) = 0  (momentum conservation)
//! ∂E/∂t + ∇·((E+p)u) = 0  (energy conservation)

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// 2D Euler equation solver for compressible inviscid flow.
#[derive(Clone, Serialize, Deserialize)]
pub struct EulerSolver {
    /// Density field [ny * nx]
    pub density: DVector<f64>,
    /// x-momentum (ρu) [ny * nx]
    pub momentum_x: DVector<f64>,
    /// y-momentum (ρv) [ny * nx]
    pub momentum_y: DVector<f64>,
    /// Total energy per unit volume [ny * nx]
    pub energy: DVector<f64>,
    /// Adiabatic index (γ = 1.4 for air)
    pub gamma: f64,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub length_x: f64,
    pub length_y: f64,
    pub time: f64,
}

impl EulerSolver {
    /// Create a new Euler solver.
    pub fn new(nx: usize, ny: usize, gamma: f64, length_x: f64, length_y: f64) -> Self {
        let n = nx * ny;
        Self {
            density: DVector::from_element(n, 1.0),
            momentum_x: DVector::zeros(n),
            momentum_y: DVector::zeros(n),
            energy: DVector::zeros(n),
            gamma,
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

    /// Get velocity components from conservative variables.
    pub fn velocity_x(&self) -> DVector<f64> {
        self.momentum_x.component_div(&self.density)
    }

    pub fn velocity_y(&self) -> DVector<f64> {
        self.momentum_y.component_div(&self.density)
    }

    /// Compute pressure from conservative variables: p = (γ-1)(E - ½ρ(u²+v²)).
    pub fn pressure(&self) -> DVector<f64> {
        let mut p = DVector::zeros(self.density.len());
        for i in 0..self.density.len() {
            let rho = self.density[i];
            let u = self.momentum_x[i] / rho;
            let v = self.momentum_y[i] / rho;
            p[i] = (self.gamma - 1.0) * (self.energy[i] - 0.5 * rho * (u * u + v * v));
        }
        p
    }

    /// Initialize with uniform flow at given conditions.
    pub fn init_uniform(&mut self, rho: f64, u: f64, v: f64, p: f64) {
        for k in 0..self.density.len() {
            self.density[k] = rho;
            self.momentum_x[k] = rho * u;
            self.momentum_y[k] = rho * v;
            self.energy[k] = p / (self.gamma - 1.0) + 0.5 * rho * (u * u + v * v);
        }
    }

    /// Initialize a Sod shock tube (1D, placed along x-axis at mid-domain).
    pub fn init_sod_shock_tube(&mut self, rho_left: f64, p_left: f64, rho_right: f64, p_right: f64) {
        let mid = self.nx / 2;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let k = self.idx(i, j);
                let (rho, p) = if i < mid {
                    (rho_left, p_left)
                } else {
                    (rho_right, p_right)
                };
                self.density[k] = rho;
                self.momentum_x[k] = 0.0;
                self.momentum_y[k] = 0.0;
                self.energy[k] = p / (self.gamma - 1.0);
            }
        }
    }

    /// Compute sound speed field.
    pub fn sound_speed(&self) -> DVector<f64> {
        let p = self.pressure();
        let mut cs = DVector::zeros(self.density.len());
        for i in 0..self.density.len() {
            let p_safe = p[i].max(1e-10);
            cs[i] = (self.gamma * p_safe / self.density[i]).sqrt();
        }
        cs
    }

    /// Compute the CFL-stable time step.
    pub fn stable_dt(&self, cfl: f64) -> f64 {
        let u = self.velocity_x();
        let v = self.velocity_y();
        let cs = self.sound_speed();
        let mut max_speed: f64 = 1e-10;
        for i in 0..self.density.len() {
            max_speed = max_speed.max((u[i].abs() + cs[i]).max(v[i].abs() + cs[i]));
        }
        cfl * self.dx.min(self.dy) / max_speed
    }

    /// Advance one step using the Lax-Friedrichs scheme.
    pub fn step(&mut self, dt: f64) {
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dy = self.dy;
        let gamma = self.gamma;

        let mut new_rho = self.density.clone();
        let mut new_mx = self.momentum_x.clone();
        let mut new_my = self.momentum_y.clone();
        let mut new_e = self.energy.clone();

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = self.idx(i, j);
                let rho = self.density[k];
                let mx = self.momentum_x[k];
                let my = self.momentum_y[k];
                let e = self.energy[k];
                let u = mx / rho;
                let v = my / rho;
                let p = (gamma - 1.0) * (e - 0.5 * rho * (u * u + v * v));
                let etot_p = e + p;

                // Lax-Friedrichs: average of neighbors
                let kl = self.idx(i - 1, j);
                let kr = self.idx(i + 1, j);
                let kb = self.idx(i, j - 1);
                let kt = self.idx(i, j + 1);

                let avg_rho = 0.25 * (self.density[kl] + self.density[kr] + self.density[kb] + self.density[kt]);
                let avg_mx = 0.25 * (self.momentum_x[kl] + self.momentum_x[kr] + self.momentum_x[kb] + self.momentum_x[kt]);
                let avg_my = 0.25 * (self.momentum_y[kl] + self.momentum_y[kr] + self.momentum_y[kb] + self.momentum_y[kt]);
                let avg_e = 0.25 * (self.energy[kl] + self.energy[kr] + self.energy[kb] + self.energy[kt]);

                // Fluxes in x-direction
                let flux_rho_x = mx;
                let flux_mx_x = mx * u + p;
                let flux_my_x = my * u;
                let flux_e_x = etot_p * u;

                // Fluxes in y-direction
                let flux_rho_y = my;
                let flux_mx_y = mx * v;
                let flux_my_y = my * v + p;
                let flux_e_y = etot_p * v;

                // Lax-Friedrichs update
                let dflux_rho_x = (self.momentum_x[kr] - self.momentum_x[kl]) / (2.0 * dx);
                let dflux_rho_y = (self.momentum_y[kt] - self.momentum_y[kb]) / (2.0 * dy);
                new_rho[k] = avg_rho - dt * (dflux_rho_x + dflux_rho_y);

                let dflux_mx_x = ((self.momentum_x[kr] * self.momentum_x[kr] / self.density[kr]
                    + (gamma - 1.0)
                        * (self.energy[kr]
                            - 0.5 * self.momentum_x[kr].powi(2) / self.density[kr]
                            - 0.5 * self.momentum_y[kr].powi(2) / self.density[kr]))
                    - (self.momentum_x[kl] * self.momentum_x[kl] / self.density[kl]
                        + (gamma - 1.0)
                            * (self.energy[kl]
                                - 0.5 * self.momentum_x[kl].powi(2) / self.density[kl]
                                - 0.5 * self.momentum_y[kl].powi(2) / self.density[kl])))
                    / (2.0 * dx);
                let dflux_mx_y = (self.momentum_x[kt] * self.momentum_y[kt] / self.density[kt]
                    - self.momentum_x[kb] * self.momentum_y[kb] / self.density[kb])
                    / (2.0 * dy);
                new_mx[k] = avg_mx - dt * (dflux_mx_x + dflux_mx_y);

                let dflux_my_x = (self.momentum_x[kr] * self.momentum_y[kr] / self.density[kr]
                    - self.momentum_x[kl] * self.momentum_y[kl] / self.density[kl])
                    / (2.0 * dx);
                let dflux_my_y = ((self.momentum_y[kt] * self.momentum_y[kt] / self.density[kt]
                    + (gamma - 1.0)
                        * (self.energy[kt]
                            - 0.5 * self.momentum_x[kt].powi(2) / self.density[kt]
                            - 0.5 * self.momentum_y[kt].powi(2) / self.density[kt]))
                    - (self.momentum_y[kb] * self.momentum_y[kb] / self.density[kb]
                        + (gamma - 1.0)
                            * (self.energy[kb]
                                - 0.5 * self.momentum_x[kb].powi(2) / self.density[kb]
                                - 0.5 * self.momentum_y[kb].powi(2) / self.density[kb])))
                    / (2.0 * dy);
                new_my[k] = avg_my - dt * (dflux_my_x + dflux_my_y);

                // Energy update (simplified)
                new_e[k] = avg_e
                    - dt * ((self.momentum_x[kr] * self.energy[kr] / self.density[kr]
                        - self.momentum_x[kl] * self.energy[kl] / self.density[kl])
                        / (2.0 * dx)
                        + (self.momentum_y[kt] * self.energy[kt] / self.density[kt]
                            - self.momentum_y[kb] * self.energy[kb] / self.density[kb])
                            / (2.0 * dy));
            }
        }

        self.density = new_rho;
        self.momentum_x = new_mx;
        self.momentum_y = new_my;
        self.energy = new_e;
        self.time += dt;
    }

    /// Advance multiple steps.
    pub fn advance(&mut self, dt: f64, steps: usize) {
        for _ in 0..steps {
            self.step(dt);
        }
    }

    /// Compute total mass (integral of density).
    pub fn total_mass(&self) -> f64 {
        self.density.iter().map(|&r| r * self.dx * self.dy).sum()
    }

    /// Compute total momentum.
    pub fn total_momentum(&self) -> (f64, f64) {
        let px: f64 = self.momentum_x.iter().map(|&m| m * self.dx * self.dy).sum();
        let py: f64 = self.momentum_y.iter().map(|&m| m * self.dx * self.dy).sum();
        (px, py)
    }

    /// Compute total energy.
    pub fn total_energy(&self) -> f64 {
        self.energy.iter().map(|&e| e * self.dx * self.dy).sum()
    }
}
