//! Vortex methods: point vortices and vortex sheets.
//!
//! Models inviscid vortex dynamics using the Biot-Savart law.

use serde::{Serialize, Deserialize};

/// A single point vortex with position, circulation, and optional core radius.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointVortex {
    /// x-position
    pub x: f64,
    /// y-position
    pub y: f64,
    /// Circulation strength (positive = counterclockwise)
    pub circulation: f64,
    /// Core radius for regularization (prevents singularity)
    pub core_radius: f64,
}

impl PointVortex {
    /// Create a new point vortex.
    pub fn new(x: f64, y: f64, circulation: f64) -> Self {
        Self {
            x,
            y,
            circulation,
            core_radius: 0.01,
        }
    }

    /// Create with a specific core radius.
    pub fn with_core(x: f64, y: f64, circulation: f64, core_radius: f64) -> Self {
        Self {
            x,
            y,
            circulation,
            core_radius,
        }
    }

    /// Compute velocity induced at point (px, py) by this vortex.
    /// Uses Lamb-Oseen regularization to avoid singularity.
    pub fn velocity_at(&self, px: f64, py: f64) -> (f64, f64) {
        let dx = px - self.x;
        let dy = py - self.y;
        let r2 = dx * dx + dy * dy + self.core_radius * self.core_radius;
        let factor = self.circulation / (2.0 * std::f64::consts::PI * r2);
        // Velocity is perpendicular to displacement
        (-factor * dy, factor * dx)
    }
}

/// A vortex sheet: a collection of point vortices along a line.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VortexSheet {
    /// The point vortices making up the sheet
    pub vortices: Vec<PointVortex>,
    /// Total circulation
    pub total_circulation: f64,
}

impl VortexSheet {
    /// Create a flat vortex sheet along the x-axis from x0 to x1 at height y,
    /// with total circulation Gamma distributed over n points.
    pub fn horizontal(x0: f64, x1: f64, y: f64, gamma: f64, n: usize) -> Self {
        let mut vortices = Vec::with_capacity(n);
        let dgamma = gamma / n as f64;
        let dx = (x1 - x0) / (n - 1).max(1) as f64;
        for i in 0..n {
            let x = x0 + i as f64 * dx;
            vortices.push(PointVortex::new(x, y, dgamma));
        }
        Self {
            vortices,
            total_circulation: gamma,
        }
    }

    /// Compute velocity at a point from all vortices in the sheet.
    pub fn velocity_at(&self, px: f64, py: f64) -> (f64, f64) {
        let mut ux = 0.0;
        let mut uy = 0.0;
        for v in &self.vortices {
            let (vx, vy) = v.velocity_at(px, py);
            ux += vx;
            uy += vy;
        }
        (ux, uy)
    }

    /// Total circulation (sum of all vortex strengths).
    pub fn total_circulation(&self) -> f64 {
        self.vortices.iter().map(|v| v.circulation).sum()
    }
}

/// A system of interacting point vortices.
#[derive(Clone, Serialize, Deserialize)]
pub struct VortexSystem {
    /// Point vortices in the system
    pub vortices: Vec<PointVortex>,
    /// Current time
    pub time: f64,
}

impl VortexSystem {
    /// Create an empty vortex system.
    pub fn new() -> Self {
        Self {
            vortices: Vec::new(),
            time: 0.0,
        }
    }

    /// Add a point vortex.
    pub fn add_vortex(&mut self, x: f64, y: f64, circulation: f64) {
        self.vortices.push(PointVortex::new(x, y, circulation));
    }

    /// Add a point vortex with custom core radius.
    pub fn add_vortex_with_core(&mut self, x: f64, y: f64, circulation: f64, core: f64) {
        self.vortices.push(PointVortex::with_core(x, y, circulation, core));
    }

    /// Compute velocity at a point from all vortices.
    pub fn velocity_at(&self, px: f64, py: f64) -> (f64, f64) {
        let mut ux = 0.0;
        let mut uy = 0.0;
        for v in &self.vortices {
            let (vx, vy) = v.velocity_at(px, py);
            ux += vx;
            uy += vy;
        }
        (ux, uy)
    }

    /// Compute velocity induced on vortex i by all other vortices.
    fn induced_velocity(&self, idx: usize) -> (f64, f64) {
        let mut ux = 0.0;
        let mut uy = 0.0;
        let vi = &self.vortices[idx];
        for (j, vj) in self.vortices.iter().enumerate() {
            if j == idx {
                continue;
            }
            let (vx, vy) = vj.velocity_at(vi.x, vi.y);
            ux += vx;
            uy += vy;
        }
        (ux, uy)
    }

    /// Advance the system by dt using RK2 (midpoint method).
    pub fn step(&mut self, dt: f64) {
        let n = self.vortices.len();
        // Store initial positions
        let x0: Vec<f64> = self.vortices.iter().map(|v| v.x).collect();
        let y0: Vec<f64> = self.vortices.iter().map(|v| v.y).collect();

        // RK2 stage 1: compute velocities at current positions
        let mut k1_x = vec![0.0; n];
        let mut k1_y = vec![0.0; n];
        for i in 0..n {
            let (ux, uy) = self.induced_velocity(i);
            k1_x[i] = ux;
            k1_y[i] = uy;
        }

        // Move to midpoint
        for i in 0..n {
            self.vortices[i].x = x0[i] + 0.5 * dt * k1_x[i];
            self.vortices[i].y = y0[i] + 0.5 * dt * k1_y[i];
        }

        // RK2 stage 2: compute velocities at midpoint
        let mut k2_x = vec![0.0; n];
        let mut k2_y = vec![0.0; n];
        for i in 0..n {
            let (ux, uy) = self.induced_velocity(i);
            k2_x[i] = ux;
            k2_y[i] = uy;
        }

        // Final update
        for i in 0..n {
            self.vortices[i].x = x0[i] + dt * k2_x[i];
            self.vortices[i].y = y0[i] + dt * k2_y[i];
        }

        self.time += dt;
    }

    /// Advance multiple steps.
    pub fn advance(&mut self, dt: f64, steps: usize) {
        for _ in 0..steps {
            self.step(dt);
        }
    }

    /// Total circulation of the system.
    pub fn total_circulation(&self) -> f64 {
        self.vortices.iter().map(|v| v.circulation).sum()
    }

    /// Compute linear impulse (first moment of vorticity).
    pub fn linear_impulse(&self) -> (f64, f64) {
        let mut px = 0.0;
        let mut py = 0.0;
        for v in &self.vortices {
            px += v.circulation * v.y;
            py -= v.circulation * v.x;
        }
        (px, py)
    }

    /// Compute angular impulse (second moment of vorticity).
    pub fn angular_impulse(&self) -> f64 {
        self.vortices
            .iter()
            .map(|v| -v.circulation * (v.x * v.x + v.y * v.y))
            .sum()
    }

    /// Compute Hamiltonian (energy) of the vortex system.
    pub fn hamiltonian(&self) -> f64 {
        let mut h = 0.0;
        for i in 0..self.vortices.len() {
            for j in (i + 1)..self.vortices.len() {
                let vi = &self.vortices[i];
                let vj = &self.vortices[j];
                let dx = vi.x - vj.x;
                let dy = vi.y - vj.y;
                let r2 = dx * dx + dy * dy + vi.core_radius * vi.core_radius;
                h -= vi.circulation * vj.circulation / (4.0 * std::f64::consts::PI) * r2.ln();
            }
        }
        h
    }

    /// Compute center of vorticity.
    pub fn center_of_vorticity(&self) -> (f64, f64) {
        let total_gamma: f64 = self.vortices.iter().map(|v| v.circulation).sum();
        if total_gamma.abs() < 1e-12 {
            return (0.0, 0.0);
        }
        let xc: f64 = self.vortices.iter().map(|v| v.circulation * v.x).sum::<f64>() / total_gamma;
        let yc: f64 = self.vortices.iter().map(|v| v.circulation * v.y).sum::<f64>() / total_gamma;
        (xc, yc)
    }
}

impl Default for VortexSystem {
    fn default() -> Self {
        Self::new()
    }
}
