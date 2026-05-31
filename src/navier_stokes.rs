//! 1D and 2D Navier-Stokes equation solvers.
//!
//! Implements finite-difference discretization of the incompressible
//! Navier-Stokes equations for viscous fluid flow.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// 1D Navier-Stokes solver (Burgers' equation as simplified model).
///
/// Solves ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
#[derive(Clone, Serialize, Deserialize)]
pub struct NavierStokes1D {
    /// Velocity field on a uniform 1D grid
    pub velocity: DVector<f64>,
    /// Kinematic viscosity
    pub viscosity: f64,
    /// Grid spacing
    pub dx: f64,
    /// Domain length
    pub length: f64,
    /// Current time
    pub time: f64,
}

impl NavierStokes1D {
    /// Create a new 1D solver with given grid points, viscosity, and domain length.
    pub fn new(n: usize, viscosity: f64, length: f64) -> Self {
        Self {
            velocity: DVector::zeros(n),
            viscosity,
            dx: length / (n - 1) as f64,
            length,
            time: 0.0,
        }
    }

    /// Initialize with a cosine velocity profile u(x,0) = A·cos(2πx/L).
    pub fn init_cosine(&mut self, amplitude: f64) {
        let n = self.velocity.len();
        for i in 0..n {
            let x = i as f64 * self.dx;
            self.velocity[i] = amplitude * (2.0 * std::f64::consts::PI * x / self.length).cos();
        }
    }

    /// Initialize with a sine profile.
    pub fn init_sine(&mut self, amplitude: f64) {
        let n = self.velocity.len();
        for i in 0..n {
            let x = i as f64 * self.dx;
            self.velocity[i] = amplitude * (2.0 * std::f64::consts::PI * x / self.length).sin();
        }
    }

    /// Compute the CFL-stable time step.
    pub fn stable_dt(&self, cfl: f64) -> f64 {
        let max_vel = self.velocity.iter().cloned().fold(0.0f64, f64::max).max(1e-10);
        let dt_adv = cfl * self.dx / max_vel;
        let dt_diff = 0.5 * self.dx * self.dx / self.viscosity.max(1e-20);
        dt_adv.min(dt_diff)
    }

    /// Advance one time step using explicit finite differences.
    pub fn step(&mut self, dt: f64) {
        let n = self.velocity.len();
        let mut new_vel = self.velocity.clone();
        let dx = self.dx;
        let nu = self.viscosity;

        for i in 1..n - 1 {
            let u = self.velocity[i];
            let u_prev = self.velocity[i - 1];
            let u_next = self.velocity[i + 1];

            // Advection: upwind scheme
            let advection = if u >= 0.0 {
                u * (u - u_prev) / dx
            } else {
                u * (u_next - u) / dx
            };

            // Diffusion: central difference
            let diffusion = nu * (u_next - 2.0 * u + u_prev) / (dx * dx);

            new_vel[i] = u + dt * (-advection + diffusion);
        }

        // Boundary conditions: zero (no-slip)
        new_vel[0] = 0.0;
        new_vel[n - 1] = 0.0;

        self.velocity = new_vel;
        self.time += dt;
    }

    /// Advance for multiple steps.
    pub fn advance(&mut self, dt: f64, steps: usize) {
        for _ in 0..steps {
            self.step(dt);
        }
    }

    /// Compute total momentum (integral of velocity).
    pub fn total_momentum(&self) -> f64 {
        self.velocity.iter().map(|&v| v * self.dx).sum()
    }

    /// Compute kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.velocity.iter().map(|&v: &f64| v * v * self.dx).sum::<f64>()
    }
}

/// 2D Navier-Stokes solver for incompressible flow on a staggered grid.
///
/// Uses the projection method (fractional step):
/// 1. Predict velocity (advection + diffusion)
/// 2. Solve pressure Poisson equation
/// 3. Correct velocity to be divergence-free
#[derive(Clone, Serialize, Deserialize)]
pub struct NavierStokes2D {
    /// u-velocity (horizontal), stored as flat vector row-major [ny * nx]
    pub u: DVector<f64>,
    /// v-velocity (vertical), stored as flat vector row-major [ny * nx]
    pub v: DVector<f64>,
    /// Pressure field
    pub pressure: DVector<f64>,
    /// Kinematic viscosity
    pub viscosity: f64,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
    /// Domain dimensions
    pub length_x: f64,
    pub length_y: f64,
    /// Current time
    pub time: f64,
}

impl NavierStokes2D {
    /// Create a new 2D solver.
    pub fn new(nx: usize, ny: usize, viscosity: f64, length_x: f64, length_y: f64) -> Self {
        let n = nx * ny;
        Self {
            u: DVector::zeros(n),
            v: DVector::zeros(n),
            pressure: DVector::zeros(n),
            viscosity,
            nx,
            ny,
            dx: length_x / (nx - 1).max(1) as f64,
            dy: length_y / (ny - 1).max(1) as f64,
            length_x,
            length_y,
            time: 0.0,
        }
    }

    /// Index helper: (i, j) -> flat index.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Initialize with uniform flow.
    pub fn init_uniform(&mut self, u0: f64, v0: f64) {
        for k in 0..self.u.len() {
            self.u[k] = u0;
            self.v[k] = v0;
        }
    }

    /// Initialize with a Taylor-Green vortex (analytical solution for verification).
    pub fn init_taylor_green(&mut self, amplitude: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = i as f64 * self.dx;
                let y = j as f64 * self.dy;
                let k = self.idx(i, j);
                self.u[k] = amplitude * (2.0 * std::f64::consts::PI * x / self.length_x).cos()
                    * (2.0 * std::f64::consts::PI * y / self.length_y).sin();
                self.v[k] = -amplitude * (2.0 * std::f64::consts::PI * x / self.length_x).sin()
                    * (2.0 * std::f64::consts::PI * y / self.length_y).cos();
            }
        }
    }

    /// Compute the divergence of the velocity field.
    pub fn divergence(&self) -> DVector<f64> {
        let n = self.nx * self.ny;
        let mut div = DVector::zeros(n);
        let dx = self.dx;
        let dy = self.dy;

        for j in 1..self.ny - 1 {
            for i in 1..self.nx - 1 {
                let k = self.idx(i, j);
                let dudx = (self.u[self.idx(i + 1, j)] - self.u[self.idx(i - 1, j)]) / (2.0 * dx);
                let dvdy = (self.v[self.idx(i, j + 1)] - self.v[self.idx(i, j - 1)]) / (2.0 * dy);
                div[k] = dudx + dvdy;
            }
        }
        div
    }

    /// Compute max divergence (should be ~0 for incompressible flow).
    pub fn max_divergence(&self) -> f64 {
        self.divergence().iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()))
    }

    /// Compute vorticity ω = ∂v/∂x - ∂u/∂y.
    pub fn vorticity(&self) -> DVector<f64> {
        let n = self.nx * self.ny;
        let mut omega = DVector::zeros(n);
        let dx = self.dx;
        let dy = self.dy;

        for j in 1..self.ny - 1 {
            for i in 1..self.nx - 1 {
                let k = self.idx(i, j);
                let dvdx = (self.v[self.idx(i + 1, j)] - self.v[self.idx(i - 1, j)]) / (2.0 * dx);
                let dudy = (self.u[self.idx(i, j + 1)] - self.u[self.idx(i, j - 1)]) / (2.0 * dy);
                omega[k] = dvdx - dudy;
            }
        }
        omega
    }

    /// Solve the pressure Poisson equation using Jacobi iteration.
    fn solve_pressure_poisson(&mut self, rhs: &DVector<f64>, iterations: usize) {
        let dx2 = self.dx * self.dx;
        let dy2 = self.dy * self.dy;
        let coeff = 2.0 * (1.0 / dx2 + 1.0 / dy2);

        for _ in 0..iterations {
            let mut p_new = self.pressure.clone();
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    let k = self.idx(i, j);
                    let p_left = self.pressure[self.idx(i - 1, j)];
                    let p_right = self.pressure[self.idx(i + 1, j)];
                    let p_bottom = self.pressure[self.idx(i, j - 1)];
                    let p_top = self.pressure[self.idx(i, j + 1)];
                    p_new[k] = ((p_right + p_left) / dx2 + (p_top + p_bottom) / dy2 - rhs[k])
                        / coeff;
                }
            }
            // Neumann BC (zero gradient)
            for i in 0..self.nx {
                p_new[self.idx(i, 0)] = p_new[self.idx(i, 1)];
                p_new[self.idx(i, self.ny - 1)] = p_new[self.idx(i, self.ny - 2)];
            }
            for j in 0..self.ny {
                p_new[self.idx(0, j)] = p_new[self.idx(1, j)];
                p_new[self.idx(self.nx - 1, j)] = p_new[self.idx(self.nx - 2, j)];
            }
            self.pressure = p_new;
        }
    }

    /// Advance one time step using the projection method.
    pub fn step(&mut self, dt: f64, pressure_iters: usize) {
        let dx = self.dx;
        let dy = self.dy;
        let nu = self.viscosity;
        let nx = self.nx;
        let ny = self.ny;

        let mut u_star = self.u.clone();
        let mut v_star = self.v.clone();

        // Step 1: Predict velocity (advection + diffusion, no pressure gradient)
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = self.idx(i, j);
                let u_ij = self.u[k];
                let v_ij = self.v[k];

                // Central differences for advection
                let dudx = (self.u[self.idx(i + 1, j)] - self.u[self.idx(i - 1, j)]) / (2.0 * dx);
                let dudy = (self.u[self.idx(i, j + 1)] - self.u[self.idx(i, j - 1)]) / (2.0 * dy);
                let dvdx = (self.v[self.idx(i + 1, j)] - self.v[self.idx(i - 1, j)]) / (2.0 * dx);
                let dvdy = (self.v[self.idx(i, j + 1)] - self.v[self.idx(i, j - 1)]) / (2.0 * dy);

                // Laplacian for diffusion
                let lap_u = (self.u[self.idx(i + 1, j)] - 2.0 * u_ij + self.u[self.idx(i - 1, j)])
                    / (dx * dx)
                    + (self.u[self.idx(i, j + 1)] - 2.0 * u_ij + self.u[self.idx(i, j - 1)])
                    / (dy * dy);
                let lap_v = (self.v[self.idx(i + 1, j)] - 2.0 * v_ij + self.v[self.idx(i - 1, j)])
                    / (dx * dx)
                    + (self.v[self.idx(i, j + 1)] - 2.0 * v_ij + self.v[self.idx(i, j - 1)])
                    / (dy * dy);

                u_star[k] = u_ij + dt * (-u_ij * dudx - v_ij * dudy + nu * lap_u);
                v_star[k] = v_ij + dt * (-u_ij * dvdx - v_ij * dvdy + nu * lap_v);
            }
        }

        // Step 2: Build RHS for pressure Poisson equation: ∇²p = (1/dt)·∇·u*
        let mut rhs = DVector::zeros(nx * ny);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = self.idx(i, j);
                let div_ustar = (u_star[self.idx(i + 1, j)] - u_star[self.idx(i - 1, j)])
                    / (2.0 * dx)
                    + (v_star[self.idx(i, j + 1)] - v_star[self.idx(i, j - 1)]) / (2.0 * dy);
                rhs[k] = div_ustar / dt;
            }
        }

        // Step 3: Solve for pressure
        self.solve_pressure_poisson(&rhs, pressure_iters);

        // Step 4: Correct velocity
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = self.idx(i, j);
                let dpdx =
                    (self.pressure[self.idx(i + 1, j)] - self.pressure[self.idx(i - 1, j)])
                    / (2.0 * dx);
                let dpdy =
                    (self.pressure[self.idx(i, j + 1)] - self.pressure[self.idx(i, j - 1)])
                    / (2.0 * dy);
                self.u[k] = u_star[k] - dt * dpdx;
                self.v[k] = v_star[k] - dt * dpdy;
            }
        }

        self.time += dt;
    }

    /// Advance for multiple steps.
    pub fn advance(&mut self, dt: f64, steps: usize, pressure_iters: usize) {
        for _ in 0..steps {
            self.step(dt, pressure_iters);
        }
    }

    /// Compute total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        let dx = self.dx * self.dy;
        0.5 * self.u.iter().zip(self.v.iter()).map(|(&u, &v)| (u * u + v * v) * dx).sum::<f64>()
    }

    /// Compute total mass (average density assuming unit density).
    pub fn total_mass(&self) -> f64 {
        (self.u.len() as f64) * self.dx * self.dy
    }
}
