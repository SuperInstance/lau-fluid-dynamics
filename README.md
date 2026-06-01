# lau-fluid-dynamics

**Computational fluid dynamics in Rust: Navier-Stokes, Euler, Lattice Boltzmann (D2Q9), vortex methods, advection-diffusion, potential flow, and agent-based flow dynamics.**

A from-scratch CFD library that solves the fundamental equations of fluid mechanics on structured grids. Nine modules, each a self-contained solver or model, all serializable and composable.

---

## What This Does

| Module | Solver | Equation / Method |
|---|---|---|
| `navier_stokes` | `NavierStokes1D`, `NavierStokes2D` | Incompressible Navier-Stokes (projection method) |
| `euler` | `EulerSolver` | 2D compressible Euler (Lax-Friedrichs) |
| `lattice_boltzmann` | `LatticeBoltzmannD2Q9` | D2Q9 lattice Boltzmann (BGK collision) |
| `advection_diffusion` | `AdvectionDiffusionSolver` | Advection-diffusion with SUPG stabilization |
| `vortex` | `PointVortex`, `VortexSheet`, `VortexSystem` | Biot-Savart vortex dynamics (RK2) |
| `potential` | `StreamFunction`, `VelocityPotential` | Potential flow on a grid |
| `flow_regime` | `ReynoldsNumber`, `FlowRegime` | Re computation and regime classification |
| `cavity` | `LidDrivenCavity` | Classic lid-driven cavity benchmark |
| `agent_flow` | `AgentFlowModel` | Agent populations modeled as fluids |

---

## Key Idea

Fluid dynamics isn't just for fluids. The same PDEs that govern airflow over a wing also describe crowd dynamics, network traffic, and agent populations. This library solves those PDEs with straightforward finite-difference and lattice methods, making CFD accessible as a Rust library rather than a monolithic Fortran codebase.

---

## Install

```toml
[dependencies]
lau-fluid-dynamics = "0.1"
```

Or:

```sh
cargo add lau-fluid-dynamics
```

Dependencies: `nalgebra` (with serde) and `serde`. Rust 2021 edition.

---

## Quick Start

```rust
use lau_fluid_dynamics::*;

// --- 1D Navier-Stokes (Burgers' equation) ---
let mut ns = NavierStokes1D::new(100, 0.01, 1.0); // 100 points, ν=0.01, L=1.0
ns.init_sine(1.0);
let dt = ns.stable_dt(0.5);
ns.advance(dt, 100);
println!("KE = {}", ns.kinetic_energy());

// --- 2D Navier-Stokes with Taylor-Green vortex ---
let mut ns2d = NavierStokes2D::new(32, 32, 0.01, 1.0, 1.0);
ns2d.init_taylor_green(1.0);
ns2d.advance(0.001, 100, 50); // dt, steps, pressure iterations
println!("Max divergence = {}", ns2d.max_divergence());

// --- Lattice Boltzmann D2Q9 ---
let mut lbm = LatticeBoltzmannD2Q9::new(50, 50, 0.8); // τ=0.8
lbm.init_equilibrium(1.0, 0.1, 0.0);
lbm.add_rectangular_obstacle(20, 20, 30, 30); // block in center
lbm.advance(200);
println!("KE = {}", lbm.kinetic_energy());

// --- Vortex dynamics ---
let mut sys = VortexSystem::new();
sys.add_vortex(0.0, 0.5, 1.0);  // positive circulation
sys.add_vortex(0.0, -0.5, -1.0); // negative circulation
println!("Hamiltonian = {}", sys.hamiltonian());
sys.advance(0.01, 50);

// --- Reynolds number & flow regime ---
let re = ReynoldsNumber::new(10.0, 1.0, 0.01);
println!("Re = {}, regime = {}", re.re, re.regime());
println!("Friction factor = {}", re.friction_factor_pipe(0.001));

// --- Lid-driven cavity ---
let mut cavity = LidDrivenCavity::new(32, 100.0, 1.0); // Re=100
let dt = cavity.stable_dt(0.3);
cavity.advance(dt, 500, 50);
let (vx, vy, omega) = cavity.primary_vortex();
println!("Primary vortex at ({:.2}, {:.2}), ω={:.3}", vx, vy, omega);

// --- Advection-diffusion ---
let mut ad = AdvectionDiffusionSolver::new(50, 50, 0.1, 1.0, 1.0);
ad.set_uniform_velocity(0.5, 0.0);
ad.init_gaussian(0.25, 0.5, 0.05, 1.0);
let dt = ad.stable_dt(0.5);
ad.advance(dt, 100);

// --- Potential flow ---
let mut vp = VelocityPotential::new(50, 50, 1.0, 1.0);
vp.init_source(0.5, 0.5, 1.0);
let (u, v) = vp.velocity();

// --- Stream function ---
let mut sf = StreamFunction::new(50, 50, 1.0, 1.0);
sf.init_vortex(0.5, 0.5, 5.0, 0.1);

// --- Agent flow model ---
let params = AgentFlowParams {
    agent_count: 100.0, agent_speed: 1.0,
    domain_size: 10.0, interaction_strength: 0.5,
};
let mut model = AgentFlowModel::new(32, params);
model.init_uniform(1.0, 0.5, 0.0);
model.advance(0.01, 50, 20);
let summary = model.summary();
println!("Regime: {}, density variance: {:.4}", summary.regime, summary.density_variance);
```

---

## API Reference

### `NavierStokes1D` — Burgers' Equation

1D viscous Burgers' equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

| Method | Description |
|---|---|
| `new(n, viscosity, length)` | Grid with `n` points |
| `init_cosine(amplitude)` / `init_sine(amplitude)` | Initial conditions |
| `stable_dt(cfl)` | CFL-based time step |
| `step(dt)` / `advance(dt, steps)` | Time integration |
| `total_momentum()` / `kinetic_energy()` | Diagnostics |

Fields: `velocity: DVector<f64>`, `viscosity`, `dx`, `length`, `time`.

### `NavierStokes2D` — Incompressible 2D

Projection method (fractional step): predict → pressure Poisson → correct.

| Method | Description |
|---|---|
| `new(nx, ny, viscosity, length_x, length_y)` | Constructor |
| `init_uniform(u0, v0)` / `init_taylor_green(amplitude)` | Initial conditions |
| `step(dt, pressure_iters)` / `advance(dt, steps, pressure_iters)` | Time integration |
| `divergence()` / `max_divergence()` / `vorticity()` | Field diagnostics |
| `kinetic_energy()` / `total_mass()` | Scalar diagnostics |

Fields: `u`, `v`, `pressure` (all `DVector<f64>`), `nx`, `ny`, `dx`, `dy`, `time`.

### `EulerSolver` — Compressible Inviscid 2D

2D Euler equations with Lax-Friedrichs scheme. Conservative variables: (ρ, ρu, ρv, E).

| Method | Description |
|---|---|
| `new(nx, ny, gamma, length_x, length_y)` | Constructor (γ=1.4 for air) |
| `init_uniform(rho, u, v, p)` / `init_sod_shock_tube(...)` | Initial conditions |
| `velocity_x()` / `velocity_y()` / `pressure()` / `sound_speed()` | Derived fields |
| `stable_dt(cfl)` | CFL condition |
| `step(dt)` / `advance(dt, steps)` | Time integration |
| `total_mass()` / `total_momentum()` / `total_energy()` | Conservation checks |

### `LatticeBoltzmannD2Q9` — D2Q9 LBM

BGK collision operator with bounce-back boundaries and solid obstacle support.

| Method | Description |
|---|---|
| `new(nx, ny, tau)` | Constructor; ν = (τ − 0.5)/3 |
| `init_equilibrium(rho0, ux, uy)` | Initial distribution |
| `viscosity()` / `set_viscosity(nu)` / `reynolds_number(L, U)` | Parameter helpers |
| `step()` / `advance(steps)` | Collision + streaming |
| `density_at(i, j)` / `velocity_at(i, j)` | Point queries |
| `density_field()` / `velocity_fields()` | Full field extraction |
| `total_mass()` / `kinetic_energy()` | Diagnostics |
| `add_rectangular_obstacle(x0, y0, x1, y1)` | Solid blocks |

Constants: `D2Q9_VELOCITIES`, `D2Q9_WEIGHTS`, `D2Q9_OPPOSITE`.

### `AdvectionDiffusionSolver` — Scalar Transport

∂φ/∂t + **u**·∇φ = D·∇²φ with upwind advection + central diffusion + SUPG stabilization.

| Method | Description |
|---|---|
| `new(nx, ny, diffusivity, length_x, length_y)` | Constructor |
| `set_uniform_velocity(u, v)` | Set velocity field |
| `init_gaussian(cx, cy, sigma, amplitude)` | Gaussian blob |
| `peclet_number(L)` / `stable_dt(cfl)` | Diagnostics |
| `step(dt)` / `advance(dt, steps)` | Time integration |
| `total_mass()` / `max_phi()` / `min_phi()` | Field stats |

### `PointVortex` / `VortexSheet` / `VortexSystem`

Biot-Savart vortex dynamics with Lamb-Oseen regularization.

| Type | Key Methods |
|---|---|
| `PointVortex` | `new(x, y, circulation)`, `with_core(...)`, `velocity_at(px, py)` |
| `VortexSheet` | `horizontal(x0, x1, y, gamma, n)`, `velocity_at(px, py)`, `total_circulation()` |
| `VortexSystem` | `add_vortex(...)`, `step(dt)` (RK2), `total_circulation()`, `linear_impulse()`, `angular_impulse()`, `hamiltonian()`, `center_of_vorticity()` |

### `StreamFunction` / `VelocityPotential`

Grid-based potential flow representations.

| Type | Key Methods |
|---|---|
| `StreamFunction` | `from_velocity_u(...)`, `velocity()`, `vorticity()`, `flow_rate(j1, j2)`, `init_uniform(U)`, `init_vortex(xc, yc, Γ, U∞)` |
| `VelocityPotential` | `velocity()`, `laplacian()`, `init_uniform(u, v)`, `init_source(xs, ys, m)` |

### `ReynoldsNumber` / `FlowRegime`

| Method / Variant | Description |
|---|---|
| `ReynoldsNumber::new(velocity, length, viscosity)` | Compute Re |
| `.regime()` | Returns `FlowRegime` enum |
| `.is_laminar()` / `.is_turbulent()` | Quick checks |
| `.friction_factor_pipe(roughness)` | Swamee-Jain approximation |
| `FlowRegime::Creeping` / `Laminar` / `Transitional` / `Turbulent` | Regime variants |

### `LidDrivenCavity`

| Method | Description |
|---|---|
| `new(n, reynolds, lid_velocity)` | Unit square cavity |
| `step(dt, pressure_iters)` / `advance(dt, steps, pressure_iters)` | Time integration |
| `stable_dt(cfl)` | CFL time step |
| `vertical_centerline_u()` / `horizontal_centerline_v()` | Benchmark profiles |
| `primary_vortex()` | (x, y, ω) of strongest vortex |
| `velocity_magnitude()` | Full magnitude field |

### `AgentFlowModel`

| Method | Description |
|---|---|
| `new(grid_size, params)` | Create with `AgentFlowParams` |
| `init_uniform(density, ux, uy)` / `init_clustered(cx, cy, sigma)` | Initial conditions |
| `behavior_regime()` | Returns `FlowRegime` |
| `detect_clusters(threshold)` | High-density regions |
| `flux_through_column(i)` | Agent flux |
| `step(dt, pressure_iters)` / `advance(...)` | Time integration |
| `summary()` | `AgentFlowSummary` struct |

---

## How It Works

### Navier-Stokes 2D (Projection Method)

1. **Predict**: Advance velocity using advection (central differences) and diffusion (Laplacian), ignoring pressure
2. **Pressure Poisson**: Solve ∇²p = (1/dt)·∇·**u**\* using Jacobi iteration
3. **Correct**: Subtract pressure gradient from predicted velocity → divergence-free field

### Lattice Boltzmann D2Q9

Particles live on a 2D lattice with 9 discrete velocity directions. Each time step:
1. **Collision**: Relax distribution toward local equilibrium via BGK: f_i → f_i − (f_i − f_i^eq)/τ
2. **Streaming**: Move distributions along their velocity directions
3. **Bounce-back**: Reverse distributions at boundaries and solid nodes

The equilibrium distribution encodes the Navier-Stokes equations in the low-Mach limit.

### Euler Equations (Lax-Friedrichs)

A first-order scheme that replaces the central value with the average of its neighbors, providing numerical diffusion for stability. Conservative form ensures shock-capturing.

### Vortex Methods

Each point vortex induces velocity via the Biot-Savart law: **u** = Γ/(2πr²) × (−dy, dx). The Lamb-Oseen core radius regularizes the singularity at r=0. Time integration uses RK2 (midpoint method). The Hamiltonian (pairwise interaction energy) and impulses (moments of vorticity) are conserved quantities for verification.

### Advection-Diffusion with SUPG

Upwind differencing for advection prevents oscillations. Central differencing for diffusion is second-order accurate. SUPG (Streamline Upwind Petrov-Galerkin) stabilization adds residual-based artificial diffusion along streamlines, controlled by the local Peclet number.

---

## The Math

### Incompressible Navier-Stokes

∂**u**/∂t + (**u**·∇)**u** = −∇p/ρ + ν∇²**u**, ∇·**u** = 0

### Burgers' Equation (1D simplification)

∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

### Euler Equations (conservation form)

∂ρ/∂t + ∇·(ρ**u**) = 0, ∂(ρ**u**)/∂t + ∇·(ρ**u**⊗**u** + p**I**) = 0, ∂E/∂t + ∇·((E+p)**u**) = 0

Pressure: p = (γ−1)(E − ½ρ|**u**|²). Sound speed: c = √(γp/ρ).

### Lattice Boltzmann (BGK)

f_i(**x** + **c**_i dt, t + dt) = f_i(**x**, t) − [f_i − f_i^eq]/τ

f_i^eq = w_i ρ [1 + 3(**c**_i·**u**) + 9/2(**c**_i·**u**)² − 3/2|**u**|²]

Kinematic viscosity: ν = (τ − 0.5)/3

### Biot-Savart Law

**u**(x) = Σ Γ_k/(2π) × (−(y−y_k), (x−x_k)) / (|**x**−**x**_k|² + σ²)

### Advection-Diffusion

∂φ/∂t + **u**·∇φ = D∇²φ, Peclet number: Pe = UL/D

### Reynolds Number

Re = UL/ν. Laminar: Re < 2300, Turbulent: Re > 4000.

### Stream Function & Velocity Potential

Stream function: u = ∂ψ/∂y, v = −∂ψ/∂x (ensures ∇·**u** = 0)
Velocity potential: u = ∂φ/∂x, v = ∂φ/∂y (irrotational flow)

---

## Testing

73 tests across unit and integration tests covering:

- 1D and 2D Navier-Stokes: initialization, conservation, divergence-free verification
- Euler: Sod shock tube, conservation laws, sound speed
- Lattice Boltzmann: equilibrium, mass conservation, obstacle interaction
- Advection-diffusion: Gaussian transport, mass conservation, Peclet number
- Vortex methods: induced velocity, conservation of Hamiltonian, circulation, impulses
- Potential flow: velocity reconstruction, Laplacian verification
- Flow regime: Re classification, friction factor
- Lid-driven cavity: boundary conditions, centerline profiles, vortex detection
- Agent flow: initialization, clustering, regime classification, flux

Run them:

```sh
cargo test
```

---

## License

MIT
