//! Lattice Boltzmann Method (LBM) with D2Q9 model.
//!
//! The D2Q9 model uses 9 velocity directions on a 2D lattice:
//!   6 2 5
//!   3 0 1
//!   7 4 8
//!
//! Weights: w0=4/9, w1-4=1/9, w5-8=1/36

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// D2Q9 lattice velocities: (cx, cy) for each of the 9 directions.
pub const D2Q9_VELOCITIES: [(i32, i32); 9] = [
    (0, 0),   // 0: rest
    (1, 0),   // 1: east
    (0, 1),   // 2: north
    (-1, 0),  // 3: west
    (0, -1),  // 4: south
    (1, 1),   // 5: NE
    (-1, 1),  // 6: NW
    (-1, -1), // 7: SW
    (1, -1),  // 8: SE
];

/// D2Q9 weights.
pub const D2Q9_WEIGHTS: [f64; 9] = [
    4.0 / 9.0,  // 0
    1.0 / 9.0,  // 1
    1.0 / 9.0,  // 2
    1.0 / 9.0,  // 3
    1.0 / 9.0,  // 4
    1.0 / 36.0, // 5
    1.0 / 36.0, // 6
    1.0 / 36.0, // 7
    1.0 / 36.0, // 8
];

/// Opposite direction indices for bounce-back boundary conditions.
pub const D2Q9_OPPOSITE: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Lattice Boltzmann D2Q9 solver.
#[derive(Clone, Serialize, Deserialize)]
pub struct LatticeBoltzmannD2Q9 {
    /// Distribution functions f_i for each direction, stored as [9 * ny * nx]
    pub f: Vec<f64>,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    /// Relaxation parameter (τ). Related to viscosity: ν = (τ - 0.5) / 3
    pub tau: f64,
    /// Solid obstacle mask (true = solid node)
    pub solid: Vec<bool>,
    /// Current time step
    pub step_count: u64,
}

impl LatticeBoltzmannD2Q9 {
    /// Create a new D2Q9 solver with the given grid size and relaxation parameter.
    pub fn new(nx: usize, ny: usize, tau: f64) -> Self {
        let n = 9 * nx * ny;
        Self {
            f: vec![0.0; n],
            nx,
            ny,
            tau,
            solid: vec![false; nx * ny],
            step_count: 0,
        }
    }

    /// Get kinematic viscosity from tau: ν = (τ - 0.5)/3.
    pub fn viscosity(&self) -> f64 {
        (self.tau - 0.5) / 3.0
    }

    /// Set tau from desired viscosity.
    pub fn set_viscosity(&mut self, nu: f64) {
        self.tau = 3.0 * nu + 0.5;
    }

    /// Compute Reynolds number.
    pub fn reynolds_number(&self, characteristic_length: f64, characteristic_velocity: f64) -> f64 {
        characteristic_length * characteristic_velocity / self.viscosity()
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    #[inline]
    fn fidx(&self, dir: usize, i: usize, j: usize) -> usize {
        dir * self.nx * self.ny + j * self.nx + i
    }

    /// Initialize with equilibrium distribution at rest (uniform density).
    pub fn init_equilibrium(&mut self, rho0: f64, ux: f64, uy: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                for k in 0..9 {
                    let val = Self::feq(k, rho0, ux, uy);
                    let idx = k * self.nx * self.ny + j * self.nx + i;
                    self.f[idx] = val;
                }
            }
        }
    }

    /// Compute equilibrium distribution for direction k.
    pub fn feq(k: usize, rho: f64, ux: f64, uy: f64) -> f64 {
        let (cx, cy) = D2Q9_VELOCITIES[k];
        let w = D2Q9_WEIGHTS[k];
        let cu = cx as f64 * ux + cy as f64 * uy;
        let usq = ux * ux + uy * uy;
        w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    }

    /// Compute macroscopic density at a node.
    pub fn density_at(&self, i: usize, j: usize) -> f64 {
        let mut rho = 0.0;
        for k in 0..9 {
            rho += self.f[self.fidx(k, i, j)];
        }
        rho
    }

    /// Compute macroscopic velocity at a node.
    pub fn velocity_at(&self, i: usize, j: usize) -> (f64, f64) {
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        for k in 0..9 {
            let fk = self.f[self.fidx(k, i, j)];
            rho += fk;
            ux += D2Q9_VELOCITIES[k].0 as f64 * fk;
            uy += D2Q9_VELOCITIES[k].1 as f64 * fk;
        }
        if rho > 1e-10 {
            (ux / rho, uy / rho)
        } else {
            (0.0, 0.0)
        }
    }

    /// Compute full density field.
    pub fn density_field(&self) -> DVector<f64> {
        let n = self.nx * self.ny;
        let mut rho = DVector::zeros(n);
        for j in 0..self.ny {
            for i in 0..self.nx {
                rho[self.idx(i, j)] = self.density_at(i, j);
            }
        }
        rho
    }

    /// Compute full velocity fields.
    pub fn velocity_fields(&self) -> (DVector<f64>, DVector<f64>) {
        let n = self.nx * self.ny;
        let mut ux = DVector::zeros(n);
        let mut uy = DVector::zeros(n);
        for j in 0..self.ny {
            for i in 0..self.nx {
                let (u, v) = self.velocity_at(i, j);
                ux[self.idx(i, j)] = u;
                uy[self.idx(i, j)] = v;
            }
        }
        (ux, uy)
    }

    /// Perform one collision + streaming step.
    pub fn step(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let tau = self.tau;

        // Collision: BGK operator f_i = f_i - (f_i - f_i^eq) / tau
        let mut f_new = self.f.clone();
        for j in 0..ny {
            for i in 0..nx {
                if self.solid[self.idx(i, j)] {
                    continue;
                }
                let rho = self.density_at(i, j);
                let (ux, uy) = self.velocity_at(i, j);

                for k in 0..9 {
                    let fi = self.f[self.fidx(k, i, j)];
                    let feq = Self::feq(k, rho, ux, uy);
                    f_new[self.fidx(k, i, j)] = fi - (fi - feq) / tau;
                }
            }
        }

        // Streaming: f_i(x + c_i, t+1) = f_i(x, t) (post-collision)
        // With bounce-back at domain boundaries
        let mut f_streamed = vec![0.0; self.f.len()];
        for j in 0..ny {
            for i in 0..nx {
                for k in 0..9 {
                    let (cx, cy) = D2Q9_VELOCITIES[k];
                    let ni = i as i32 + cx;
                    let nj = j as i32 + cy;
                    if ni >= 0 && ni < nx as i32 && nj >= 0 && nj < ny as i32 {
                        f_streamed[k * nx * ny + nj as usize * nx + ni as usize] =
                            f_new[k * nx * ny + j * nx + i];
                    } else {
                        // Bounce-back at domain boundaries
                        let opp = D2Q9_OPPOSITE[k];
                        f_streamed[opp * nx * ny + j * nx + i] =
                            f_new[k * nx * ny + j * nx + i];
                    }
                }
            }
        }

        // Bounce-back for solid nodes
        for j in 0..ny {
            for i in 0..nx {
                if self.solid[self.idx(i, j)] {
                    for k in 0..9 {
                        let opp = D2Q9_OPPOSITE[k];
                        f_streamed[self.fidx(k, i, j)] = f_new[self.fidx(opp, i, j)];
                    }
                }
            }
        }

        self.f = f_streamed;
        self.step_count += 1;
    }

    /// Advance multiple steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Compute total mass (sum of all densities).
    pub fn total_mass(&self) -> f64 {
        let rho = self.density_field();
        rho.iter().sum()
    }

    /// Compute total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        let (ux, uy) = self.velocity_fields();
        let rho = self.density_field();
        let mut ke = 0.0;
        for i in 0..rho.len() {
            ke += 0.5 * rho[i] * (ux[i] * ux[i] + uy[i] * uy[i]);
        }
        ke
    }

    /// Add a rectangular obstacle.
    pub fn add_rectangular_obstacle(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        for j in y0..y1.min(self.ny) {
            for i in x0..x1.min(self.nx) {
                let idx = self.idx(i, j);
                self.solid[idx] = true;
            }
        }
    }
}
