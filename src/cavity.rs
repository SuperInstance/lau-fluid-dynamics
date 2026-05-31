//! Lid-driven cavity flow — the classic CFD benchmark.
//!
//! A square cavity with a moving lid (top wall) drives a recirculating flow.
//! This tests the Navier-Stokes solver at various Reynolds numbers.

use crate::navier_stokes::NavierStokes2D;
use serde::{Serialize, Deserialize};

/// Lid-driven cavity flow solver.
#[derive(Clone, Serialize, Deserialize)]
pub struct LidDrivenCavity {
    /// Underlying Navier-Stokes solver
    pub solver: NavierStokes2D,
    /// Lid velocity (top wall moves at this speed)
    pub lid_velocity: f64,
    /// Reynolds number
    pub reynolds: f64,
}

impl LidDrivenCavity {
    /// Create a new cavity solver with the given grid size and Reynolds number.
    ///
    /// The cavity is a unit square [0,1] x [0,1].
    pub fn new(n: usize, reynolds: f64, lid_velocity: f64) -> Self {
        let cavity_size = 1.0;
        // ν = U * L / Re
        let viscosity = lid_velocity * cavity_size / reynolds;
        let solver = NavierStokes2D::new(n, n, viscosity, cavity_size, cavity_size);

        Self {
            solver,
            lid_velocity,
            reynolds,
        }
    }

    /// Set the lid velocity boundary condition on the top row.
    pub fn apply_boundary_conditions(&mut self) {
        let nx = self.solver.nx;
        let ny = self.solver.ny;

        // Top wall: moving lid
        for i in 0..nx {
            let k = self.solver.idx(i, ny - 1);
            self.solver.u[k] = self.lid_velocity;
            self.solver.v[k] = 0.0;
        }

        // Bottom wall: no-slip
        for i in 0..nx {
            let k = self.solver.idx(i, 0);
            self.solver.u[k] = 0.0;
            self.solver.v[k] = 0.0;
        }

        // Left wall: no-slip
        for j in 0..ny {
            let k = self.solver.idx(0, j);
            self.solver.u[k] = 0.0;
            self.solver.v[k] = 0.0;
        }

        // Right wall: no-slip
        for j in 0..ny {
            let k = self.solver.idx(nx - 1, j);
            self.solver.u[k] = 0.0;
            self.solver.v[k] = 0.0;
        }
    }

    /// Advance one step.
    pub fn step(&mut self, dt: f64, pressure_iters: usize) {
        self.solver.step(dt, pressure_iters);
        self.apply_boundary_conditions();
    }

    /// Advance for multiple steps.
    pub fn advance(&mut self, dt: f64, steps: usize, pressure_iters: usize) {
        for _ in 0..steps {
            self.step(dt, pressure_iters);
        }
    }

    /// Compute a stable time step based on CFL condition.
    pub fn stable_dt(&self, cfl: f64) -> f64 {
        let dx = self.solver.dx;
        let max_u = self.lid_velocity;
        let nu = self.solver.viscosity;

        let dt_adv = cfl * dx / max_u;
        let dt_diff = 0.25 * dx * dx / nu.max(1e-20);
        dt_adv.min(dt_diff)
    }

    /// Extract the velocity along the vertical centerline (for benchmarking).
    pub fn vertical_centerline_u(&self) -> Vec<f64> {
        let nx = self.solver.nx;
        let ny = self.solver.ny;
        let mid_i = nx / 2;
        let mut u_profile = Vec::with_capacity(ny);
        for j in 0..ny {
            let k = self.solver.idx(mid_i, j);
            u_profile.push(self.solver.u[k]);
        }
        u_profile
    }

    /// Extract the velocity along the horizontal centerline.
    pub fn horizontal_centerline_v(&self) -> Vec<f64> {
        let nx = self.solver.nx;
        let ny = self.solver.ny;
        let mid_j = ny / 2;
        let mut v_profile = Vec::with_capacity(nx);
        for i in 0..nx {
            let k = self.solver.idx(i, mid_j);
            v_profile.push(self.solver.v[k]);
        }
        v_profile
    }

    /// Find the location and value of the primary vortex center.
    pub fn primary_vortex(&self) -> (f64, f64, f64) {
        let nx = self.solver.nx;
        let ny = self.solver.ny;
        let omega = self.solver.vorticity();

        let mut max_vort = 0.0f64;
        let mut xi = 0;
        let mut yi = 0;

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = self.solver.idx(i, j);
                let v = omega[k].abs();
                if v > max_vort {
                    max_vort = v;
                    xi = i;
                    yi = j;
                }
            }
        }

        let x = xi as f64 * self.solver.dx;
        let y = yi as f64 * self.solver.dy;
        (x, y, omega[self.solver.idx(xi, yi)])
    }

    /// Get the velocity magnitude field.
    pub fn velocity_magnitude(&self) -> Vec<f64> {
        self.solver
            .u
            .iter()
            .zip(self.solver.v.iter())
            .map(|(&u, &v)| (u * u + v * v).sqrt())
            .collect()
    }
}
