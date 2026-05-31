//! Stream function and velocity potential for 2D incompressible flow.
//!
//! For incompressible 2D flow:
//! - Stream function ψ: u = ∂ψ/∂y, v = -∂ψ/∂x (automatically divergence-free)
//! - Velocity potential φ: u = ∂φ/∂x, v = ∂φ/∂y (irrotational flow)

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Stream function representation for 2D incompressible flow.
#[derive(Clone, Serialize, Deserialize)]
pub struct StreamFunction {
    /// ψ values on a 2D grid [ny * nx]
    pub psi: DVector<f64>,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub length_x: f64,
    pub length_y: f64,
}

impl StreamFunction {
    /// Create a zero stream function on the given grid.
    pub fn new(nx: usize, ny: usize, length_x: f64, length_y: f64) -> Self {
        Self {
            psi: DVector::zeros(nx * ny),
            nx,
            ny,
            dx: length_x / (nx - 1).max(1) as f64,
            dy: length_y / (ny - 1).max(1) as f64,
            length_x,
            length_y,
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Initialize from velocity field by integrating u = ∂ψ/∂y.
    pub fn from_velocity_u(u: &DVector<f64>, nx: usize, ny: usize, length_x: f64, length_y: f64) -> Self {
        let mut sf = Self::new(nx, ny, length_x, length_y);
        let dy = sf.dy;
        // Integrate along y: ψ(i, j+1) = ψ(i, j) + u(i, j) * dy
        for i in 0..nx {
            for j in 0..ny - 1 {
                let k = sf.idx(i, j);
                let k_next = sf.idx(i, j + 1);
                sf.psi[k_next] = sf.psi[k] + u[k] * dy;
            }
        }
        sf
    }

    /// Compute velocity components from stream function.
    pub fn velocity(&self) -> (DVector<f64>, DVector<f64>) {
        let n = self.nx * self.ny;
        let mut u = DVector::zeros(n);
        let mut v = DVector::zeros(n);

        for j in 1..self.ny - 1 {
            for i in 0..self.nx {
                let k = self.idx(i, j);
                // u = ∂ψ/∂y
                u[k] = (self.psi[self.idx(i, j + 1)] - self.psi[self.idx(i, j - 1)]) / (2.0 * self.dy);
            }
        }
        for j in 0..self.ny {
            for i in 1..self.nx - 1 {
                let k = self.idx(i, j);
                // v = -∂ψ/∂x
                v[k] = -(self.psi[self.idx(i + 1, j)] - self.psi[self.idx(i - 1, j)]) / (2.0 * self.dx);
            }
        }

        (u, v)
    }

    /// Compute vorticity: ω = -∇²ψ (for 2D incompressible flow).
    pub fn vorticity(&self) -> DVector<f64> {
        let n = self.nx * self.ny;
        let mut omega = DVector::zeros(n);

        for j in 1..self.ny - 1 {
            for i in 1..self.nx - 1 {
                let k = self.idx(i, j);
                let d2psi_dx2 = (self.psi[self.idx(i + 1, j)] - 2.0 * self.psi[k]
                    + self.psi[self.idx(i - 1, j)])
                    / (self.dx * self.dx);
                let d2psi_dy2 = (self.psi[self.idx(i, j + 1)] - 2.0 * self.psi[k]
                    + self.psi[self.idx(i, j - 1)])
                    / (self.dy * self.dy);
                omega[k] = -(d2psi_dx2 + d2psi_dy2);
            }
        }
        omega
    }

    /// Compute volume flow rate between two streamlines.
    pub fn flow_rate(&self, j1: usize, j2: usize) -> f64 {
        let mid_i = self.nx / 2;
        self.psi[self.idx(mid_i, j2)] - self.psi[self.idx(mid_i, j1)]
    }

    /// Initialize with a uniform stream: ψ = U * y.
    pub fn init_uniform(&mut self, u_inf: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let y = j as f64 * self.dy;
                let idx = self.idx(i, j);
                self.psi[idx] = u_inf * y;
            }
        }
    }

    /// Initialize with a point vortex at (xc, yc) with circulation Γ.
    /// ψ = -Γ/(2π) * ln(r) + U∞ * y
    pub fn init_vortex(&mut self, xc: f64, yc: f64, gamma: f64, u_inf: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = i as f64 * self.dx;
                let y = j as f64 * self.dy;
                let r = ((x - xc) * (x - xc) + (y - yc) * (y - yc)).sqrt().max(0.01);
                let idx = self.idx(i, j);
                self.psi[idx] =
                    -gamma / (2.0 * std::f64::consts::PI) * r.ln() + u_inf * y;
            }
        }
    }
}

/// Velocity potential for irrotational (potential) flow.
#[derive(Clone, Serialize, Deserialize)]
pub struct VelocityPotential {
    /// φ values on a 2D grid [ny * nx]
    pub phi: DVector<f64>,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub length_x: f64,
    pub length_y: f64,
}

impl VelocityPotential {
    /// Create a zero velocity potential.
    pub fn new(nx: usize, ny: usize, length_x: f64, length_y: f64) -> Self {
        Self {
            phi: DVector::zeros(nx * ny),
            nx,
            ny,
            dx: length_x / (nx - 1).max(1) as f64,
            dy: length_y / (ny - 1).max(1) as f64,
            length_x,
            length_y,
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Compute velocity from potential: u = ∂φ/∂x, v = ∂φ/∂y.
    pub fn velocity(&self) -> (DVector<f64>, DVector<f64>) {
        let n = self.nx * self.ny;
        let mut u = DVector::zeros(n);
        let mut v = DVector::zeros(n);

        for j in 0..self.ny {
            for i in 1..self.nx - 1 {
                let k = self.idx(i, j);
                u[k] = (self.phi[self.idx(i + 1, j)] - self.phi[self.idx(i - 1, j)])
                    / (2.0 * self.dx);
            }
        }
        for j in 1..self.ny - 1 {
            for i in 0..self.nx {
                let k = self.idx(i, j);
                v[k] = (self.phi[self.idx(i, j + 1)] - self.phi[self.idx(i, j - 1)])
                    / (2.0 * self.dy);
            }
        }
        (u, v)
    }

    /// Initialize with uniform flow: φ = U * x + V * y.
    pub fn init_uniform(&mut self, u_inf: f64, v_inf: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = i as f64 * self.dx;
                let y = j as f64 * self.dy;
                let idx = self.idx(i, j);
                self.phi[idx] = u_inf * x + v_inf * y;
            }
        }
    }

    /// Initialize with a source at (xs, ys) of strength m.
    /// φ = m/(2π) * ln(r)
    pub fn init_source(&mut self, xs: f64, ys: f64, m: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = i as f64 * self.dx;
                let y = j as f64 * self.dy;
                let r = ((x - xs) * (x - xs) + (y - ys) * (y - ys)).sqrt().max(0.01);
                let idx = self.idx(i, j);
                self.phi[idx] = m / (2.0 * std::f64::consts::PI) * r.ln();
            }
        }
    }

    /// Compute Laplacian ∇²φ (should be 0 for potential flow with no sources).
    pub fn laplacian(&self) -> DVector<f64> {
        let n = self.nx * self.ny;
        let mut lap = DVector::zeros(n);

        for j in 1..self.ny - 1 {
            for i in 1..self.nx - 1 {
                let k = self.idx(i, j);
                lap[k] = (self.phi[self.idx(i + 1, j)] - 2.0 * self.phi[k]
                    + self.phi[self.idx(i - 1, j)])
                    / (self.dx * self.dx)
                    + (self.phi[self.idx(i, j + 1)] - 2.0 * self.phi[k]
                        + self.phi[self.idx(i, j - 1)])
                    / (self.dy * self.dy);
            }
        }
        lap
    }
}
