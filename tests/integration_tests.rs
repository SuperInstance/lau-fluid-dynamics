#[cfg(test)]
mod tests {
    use lau_fluid_dynamics::*;
use lau_fluid_dynamics::agent_flow::AgentFlowParams;

    // === NAVIER-STOKES 1D ===

    #[test]
    fn test_ns1d_creation() {
        let ns = NavierStokes1D::new(100, 0.1, 1.0);
        assert_eq!(ns.velocity.len(), 100);
        assert!((ns.dx - 1.0 / 99.0).abs() < 1e-10);
        assert!((ns.viscosity - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_ns1d_cosine_init() {
        let mut ns = NavierStokes1D::new(50, 0.01, 1.0);
        ns.init_cosine(1.0);
        // At x=0, cos(0) = 1.0
        assert!((ns.velocity[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ns1d_stable_dt_positive() {
        let mut ns = NavierStokes1D::new(50, 0.01, 1.0);
        ns.init_sine(1.0);
        let dt = ns.stable_dt(0.5);
        assert!(dt > 0.0);
    }

    #[test]
    fn test_ns1d_diffusion_decays() {
        let mut ns = NavierStokes1D::new(101, 0.5, 1.0);
        ns.init_sine(1.0);
        let ke0 = ns.kinetic_energy();
        let dt = ns.stable_dt(0.3);
        ns.advance(dt, 200);
        let ke_final = ns.kinetic_energy();
        // Kinetic energy should decay due to viscous dissipation
        assert!(ke_final < ke0, "KE: {} -> {}", ke0, ke_final);
    }

    #[test]
    fn test_ns1d_boundary_conditions() {
        let mut ns = NavierStokes1D::new(50, 0.01, 1.0);
        ns.init_sine(2.0);
        let dt = ns.stable_dt(0.3);
        ns.advance(dt, 10);
        // No-slip BCs: endpoints should be zero
        assert!((ns.velocity[0]).abs() < 1e-10);
        assert!((ns.velocity[49]).abs() < 1e-10);
    }

    #[test]
    fn test_ns1d_kinetic_energy_positive() {
        let mut ns = NavierStokes1D::new(50, 0.01, 1.0);
        ns.init_cosine(1.0);
        assert!(ns.kinetic_energy() > 0.0);
    }

    // === NAVIER-STOKES 2D ===

    #[test]
    fn test_ns2d_creation() {
        let ns = NavierStokes2D::new(20, 20, 0.01, 1.0, 1.0);
        assert_eq!(ns.u.len(), 400);
        assert_eq!(ns.v.len(), 400);
    }

    #[test]
    fn test_ns2d_taylor_green_init() {
        let mut ns = NavierStokes2D::new(20, 20, 0.01, 1.0, 1.0);
        ns.init_taylor_green(1.0);
        // Center should have nonzero u
        let mid = 10 * 20 + 10;
        assert!(ns.u[mid].abs() > 0.0 || ns.v[mid].abs() > 0.0);
    }

    #[test]
    fn test_ns2d_divergence_initially_small() {
        let mut ns = NavierStokes2D::new(30, 30, 0.01, 1.0, 1.0);
        ns.init_taylor_green(1.0);
        // Taylor-Green is divergence-free by construction
        let max_div = ns.max_divergence();
        assert!(max_div < 0.1, "Max divergence: {}", max_div);
    }

    #[test]
    fn test_ns2d_vorticity() {
        let mut ns = NavierStokes2D::new(20, 20, 0.01, 1.0, 1.0);
        ns.init_taylor_green(1.0);
        let omega = ns.vorticity();
        // Taylor-Green vortex should have nonzero vorticity
        let max_omega = omega.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(max_omega > 0.0);
    }

    #[test]
    fn test_ns2d_step_advances_time() {
        let mut ns = NavierStokes2D::new(10, 10, 0.01, 1.0, 1.0);
        ns.init_uniform(1.0, 0.0);
        assert!((ns.time - 0.0).abs() < 1e-10);
        ns.step(0.001, 50);
        assert!(ns.time > 0.0);
    }

    #[test]
    fn test_ns2d_kinetic_energy() {
        let mut ns = NavierStokes2D::new(10, 10, 0.01, 1.0, 1.0);
        ns.init_uniform(1.0, 0.0);
        let ke = ns.kinetic_energy();
        assert!(ke > 0.0);
    }

    // === EULER EQUATIONS ===

    #[test]
    fn test_euler_creation() {
        let euler = EulerSolver::new(20, 20, 1.4, 1.0, 1.0);
        assert_eq!(euler.density.len(), 400);
        assert!((euler.gamma - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_euler_uniform_init() {
        let mut euler = EulerSolver::new(10, 10, 1.4, 1.0, 1.0);
        euler.init_uniform(1.0, 0.5, 0.0, 1.0);
        assert!((euler.density[0] - 1.0).abs() < 1e-10);
        assert!((euler.momentum_x[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_euler_pressure() {
        let mut euler = EulerSolver::new(10, 10, 1.4, 1.0, 1.0);
        euler.init_uniform(1.0, 0.0, 0.0, 1.0);
        let p = euler.pressure();
        assert!((p[0] - 1.0).abs() < 0.01, "Pressure: {}", p[0]);
    }

    #[test]
    fn test_euler_sound_speed() {
        let mut euler = EulerSolver::new(10, 10, 1.4, 1.0, 1.0);
        euler.init_uniform(1.0, 0.0, 0.0, 1.0);
        let cs = euler.sound_speed();
        let expected_cs: f64 = (1.4_f64 * 1.0 / 1.0).sqrt();
        assert!((cs[0] - expected_cs).abs() < 0.1, "cs: {}, expected: {}", cs[0], expected_cs);
    }

    #[test]
    fn test_euler_stable_dt() {
        let mut euler = EulerSolver::new(10, 10, 1.4, 1.0, 1.0);
        euler.init_uniform(1.0, 1.0, 0.0, 1.0);
        let dt = euler.stable_dt(0.5);
        assert!(dt > 0.0);
    }

    #[test]
    fn test_euler_mass_conservation() {
        let mut euler = EulerSolver::new(20, 20, 1.4, 1.0, 1.0);
        euler.init_uniform(1.0, 0.0, 0.0, 1.0);
        let m0 = euler.total_mass();
        let dt = euler.stable_dt(0.3);
        euler.advance(dt, 5);
        let m1 = euler.total_mass();
        assert!((m0 - m1).abs() / m0 < 0.1, "Mass drift: {} -> {}", m0, m1);
    }

    #[test]
    fn test_euler_sod_shock_tube() {
        let mut euler = EulerSolver::new(50, 10, 1.4, 1.0, 0.2);
        euler.init_sod_shock_tube(1.0, 1.0, 0.125, 0.1);
        // Left half should have rho=1, right half rho=0.125
        assert!((euler.density[0] - 1.0).abs() < 1e-10);
        let right_idx = 5 * 50 + 40; // valid: j=5, i=40
        assert!((euler.density[right_idx] - 0.125).abs() < 1e-10);
    }

    // === LATTICE BOLTZMANN ===

    #[test]
    fn test_lbm_creation() {
        let lbm = LatticeBoltzmannD2Q9::new(20, 20, 0.6);
        assert_eq!(lbm.f.len(), 9 * 20 * 20);
        assert!((lbm.viscosity() - (0.6 - 0.5) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lbm_equilibrium() {
        let rho = 1.0;
        let ux = 0.0;
        let uy = 0.0;
        // At rest, feq = w_i * rho
        let feq0 = LatticeBoltzmannD2Q9::feq(0, rho, ux, uy);
        assert!((feq0 - 4.0 / 9.0).abs() < 1e-10);
        let feq1 = LatticeBoltzmannD2Q9::feq(1, rho, ux, uy);
        assert!((feq1 - 1.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_lbm_init_equilibrium_mass() {
        let mut lbm = LatticeBoltzmannD2Q9::new(20, 20, 0.6);
        lbm.init_equilibrium(1.0, 0.0, 0.0);
        let mass = lbm.total_mass();
        let expected = 1.0 * 20.0 * 20.0;
        assert!((mass - expected).abs() / expected < 0.01, "Mass: {}, expected: {}", mass, expected);
    }

    #[test]
    fn test_lbm_mass_conservation() {
        let mut lbm = LatticeBoltzmannD2Q9::new(20, 20, 0.8);
        lbm.init_equilibrium(1.0, 0.0, 0.0);
        let m0 = lbm.total_mass();
        lbm.advance(5);
        let m1 = lbm.total_mass();
        // At rest with no velocity, mass should be very well conserved
        assert!((m0 - m1).abs() / m0 < 0.05, "Mass: {} -> {}", m0, m1);
    }

    #[test]
    fn test_lbm_set_viscosity() {
        let mut lbm = LatticeBoltzmannD2Q9::new(10, 10, 1.0);
        lbm.set_viscosity(0.1);
        assert!((lbm.viscosity() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_lbm_reynolds_number() {
        let lbm = LatticeBoltzmannD2Q9::new(10, 10, 0.6);
        let re = lbm.reynolds_number(10.0, 0.1);
        let expected = 10.0 * 0.1 / lbm.viscosity();
        assert!((re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lbm_velocity_at_rest() {
        let mut lbm = LatticeBoltzmannD2Q9::new(10, 10, 0.6);
        lbm.init_equilibrium(1.0, 0.0, 0.0);
        let (ux, uy) = lbm.velocity_at(5, 5);
        assert!(ux.abs() < 1e-10);
        assert!(uy.abs() < 1e-10);
    }

    #[test]
    fn test_lbm_obstacle() {
        let mut lbm = LatticeBoltzmannD2Q9::new(20, 20, 0.6);
        lbm.init_equilibrium(1.0, 0.0, 0.0);
        lbm.add_rectangular_obstacle(8, 8, 12, 12);
        assert!(lbm.solid[10 * 20 + 10]); // center of obstacle
        assert!(!lbm.solid[0]); // corner should not be solid
    }

    #[test]
    fn test_lbm_density_fields() {
        let mut lbm = LatticeBoltzmannD2Q9::new(10, 10, 0.6);
        lbm.init_equilibrium(2.0, 0.0, 0.0);
        let rho = lbm.density_field();
        assert!((rho[50] - 2.0).abs() < 0.01);
    }

    // === ADVECTION-DIFFUSION ===

    #[test]
    fn test_ad_creation() {
        let ad = AdvectionDiffusionSolver::new(20, 20, 0.1, 1.0, 1.0);
        assert_eq!(ad.phi.len(), 400);
    }

    #[test]
    fn test_ad_gaussian_init() {
        let mut ad = AdvectionDiffusionSolver::new(30, 30, 0.1, 1.0, 1.0);
        ad.init_gaussian(0.5, 0.5, 0.1, 1.0);
        // Peak should be at center
        let center = 15 * 30 + 15;
        let corner = 0;
        assert!(ad.phi[center] > ad.phi[corner]);
    }

    #[test]
    fn test_ad_diffusion_spreads() {
        let mut ad = AdvectionDiffusionSolver::new(30, 30, 0.05, 1.0, 1.0);
        ad.init_gaussian(0.5, 0.5, 0.05, 1.0);
        let max0 = ad.max_phi();
        let dt = ad.stable_dt(0.3);
        ad.advance(dt, 50);
        let max_final = ad.max_phi();
        // Peak should decrease as Gaussian diffuses
        assert!(max_final < max0);
    }

    #[test]
    fn test_ad_peclet_number() {
        let mut ad = AdvectionDiffusionSolver::new(10, 10, 0.1, 1.0, 1.0);
        ad.set_uniform_velocity(1.0, 0.0);
        let pe = ad.peclet_number(1.0);
        assert!((pe - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_ad_stable_dt() {
        let mut ad = AdvectionDiffusionSolver::new(20, 20, 0.1, 1.0, 1.0);
        ad.set_uniform_velocity(1.0, 0.5);
        let dt = ad.stable_dt(0.5);
        assert!(dt > 0.0);
    }

    #[test]
    fn test_ad_min_max() {
        let mut ad = AdvectionDiffusionSolver::new(50, 50, 0.1, 1.0, 1.0);
        ad.init_gaussian(0.5, 0.5, 0.1, 2.0);
        // Peak of Gaussian at center should be close to 2.0
        // At exact center on a 50x50 grid: x=25*0.0204=0.51, close enough
        assert!(ad.max_phi() > 1.5, "Max phi: {}", ad.max_phi());
    }

    // === VORTEX METHODS ===

    #[test]
    fn test_point_vortex_creation() {
        let v = PointVortex::new(1.0, 2.0, 3.0);
        assert!((v.x - 1.0).abs() < 1e-10);
        assert!((v.y - 2.0).abs() < 1e-10);
        assert!((v.circulation - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_vortex_velocity() {
        // Vortex at origin with Gamma = 2π → at (1, 0) should give u=(0, 1)
        let v = PointVortex::with_core(0.0, 0.0, 2.0 * std::f64::consts::PI, 0.01);
        let (ux, uy) = v.velocity_at(1.0, 0.0);
        // Velocity perpendicular to displacement, counterclockwise
        assert!(ux.abs() < 0.5);
        assert!(uy > 0.5);
    }

    #[test]
    fn test_vortex_system_co_rotation() {
        // Two equal vortices should co-rotate around their center
        let mut sys = VortexSystem::new();
        sys.add_vortex_with_core(1.0, 0.0, 2.0 * std::f64::consts::PI, 0.1);
        sys.add_vortex_with_core(-1.0, 0.0, 2.0 * std::f64::consts::PI, 0.1);
        let (xc0, yc0) = sys.center_of_vorticity();
        let dt = 0.01;
        sys.advance(dt, 10);
        let (xc1, yc1) = sys.center_of_vorticity();
        // Center of vorticity should be approximately conserved
        assert!((xc0 - xc1).abs() < 0.1, "xc: {} -> {}", xc0, xc1);
        assert!((yc0 - yc1).abs() < 0.1, "yc: {} -> {}", yc0, yc1);
    }

    #[test]
    fn test_vortex_system_circulation_conservation() {
        let mut sys = VortexSystem::new();
        sys.add_vortex(0.0, 0.0, 5.0);
        sys.add_vortex(1.0, 0.0, -3.0);
        sys.add_vortex(0.0, 1.0, 2.0);
        let gamma0 = sys.total_circulation();
        sys.advance(0.01, 20);
        let gamma1 = sys.total_circulation();
        assert!((gamma0 - gamma1).abs() < 1e-10);
    }

    #[test]
    fn test_vortex_sheet_creation() {
        let sheet = VortexSheet::horizontal(0.0, 1.0, 0.0, 10.0, 20);
        assert_eq!(sheet.vortices.len(), 20);
        let total = sheet.total_circulation();
        assert!((total - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_vortex_system_hamiltonian() {
        let mut sys = VortexSystem::new();
        sys.add_vortex_with_core(1.0, 0.0, 1.0, 0.1);
        sys.add_vortex_with_core(-1.0, 0.0, 1.0, 0.1);
        let h0 = sys.hamiltonian();
        let dt = 0.005;
        sys.advance(dt, 5);
        let h1 = sys.hamiltonian();
        // Hamiltonian should be approximately conserved
        assert!((h0 - h1).abs() / h0.abs().max(1e-10) < 0.1);
    }

    #[test]
    fn test_vortex_angular_impulse() {
        let mut sys = VortexSystem::new();
        sys.add_vortex(1.0, 0.0, 1.0);
        sys.add_vortex(-1.0, 0.0, 1.0);
        let ang = sys.angular_impulse();
        // I = -Σ Γ_i (x_i² + y_i²) = -(1*1 + 1*1) = -2
        assert!((ang - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_vortex_linear_impulse() {
        let sys = VortexSystem::new();
        // Empty system has zero impulse
        let (px, py) = sys.linear_impulse();
        assert!((px).abs() < 1e-10);
        assert!((py).abs() < 1e-10);
    }

    // === STREAM FUNCTION & VELOCITY POTENTIAL ===

    #[test]
    fn test_stream_function_uniform() {
        let mut sf = StreamFunction::new(20, 20, 1.0, 1.0);
        sf.init_uniform(1.0);
        let (u, v) = sf.velocity();
        // For uniform stream ψ = U*y → u = U = 1.0, v = 0
        let mid = 10 * 20 + 10;
        assert!((u[mid] - 1.0).abs() < 0.1, "u: {}", u[mid]);
        assert!(v[mid].abs() < 0.1, "v: {}", v[mid]);
    }

    #[test]
    fn test_stream_function_vorticity() {
        let mut sf = StreamFunction::new(30, 30, 1.0, 1.0);
        sf.init_uniform(1.0);
        let omega = sf.vorticity();
        let max_omega = omega.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        // Uniform stream has zero vorticity
        assert!(max_omega < 0.1, "Max vorticity: {}", max_omega);
    }

    #[test]
    fn test_velocity_potential_uniform() {
        let mut vp = VelocityPotential::new(20, 20, 1.0, 1.0);
        vp.init_uniform(1.0, 0.0);
        let (u, v) = vp.velocity();
        let mid = 10 * 20 + 10;
        assert!((u[mid] - 1.0).abs() < 0.1, "u: {}", u[mid]);
        assert!(v[mid].abs() < 0.1, "v: {}", v[mid]);
    }

    #[test]
    fn test_velocity_potential_laplacian() {
        let mut vp = VelocityPotential::new(20, 20, 1.0, 1.0);
        vp.init_uniform(1.0, 0.5);
        let lap = vp.laplacian();
        let max_lap = lap.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        // Uniform flow has zero Laplacian
        assert!(max_lap < 0.1, "Max Laplacian: {}", max_lap);
    }

    #[test]
    fn test_stream_function_from_velocity() {
        let nx = 20;
        let ny = 20;
        let mut u = nalgebra::DVector::from_element(nx * ny, 1.0);
        // Zero boundaries
        for i in 0..nx {
            u[i] = 0.0;
            u[(ny - 1) * nx + i] = 0.0;
        }
        let sf = StreamFunction::from_velocity_u(&u, nx, ny, 1.0, 1.0);
        // ψ should increase along y (since u = ∂ψ/∂y = 1.0)
        assert!(sf.psi[10 * nx + 10] > sf.psi[5 * nx + 10]);
    }

    // === REYNOLDS NUMBER & FLOW REGIME ===

    #[test]
    fn test_reynolds_number_calculation() {
        let re = ReynoldsNumber::new(1.0, 1.0, 0.01);
        assert!((re.re - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_flow_regime_creeping() {
        let re = ReynoldsNumber::new(0.01, 0.1, 0.01);
        assert_eq!(re.regime(), FlowRegime::Creeping);
    }

    #[test]
    fn test_flow_regime_laminar() {
        let re = ReynoldsNumber::new(0.1, 1.0, 0.01);
        assert_eq!(re.regime(), FlowRegime::Laminar);
    }

    #[test]
    fn test_flow_regime_transitional() {
        let re = ReynoldsNumber::new(3.0, 1.0, 0.001);
        assert_eq!(re.regime(), FlowRegime::Transitional);
    }

    #[test]
    fn test_flow_regime_turbulent() {
        let re = ReynoldsNumber::new(10.0, 1.0, 0.001);
        assert_eq!(re.regime(), FlowRegime::Turbulent);
    }

    #[test]
    fn test_reynolds_is_laminar() {
        let re = ReynoldsNumber::new(0.1, 1.0, 0.01);
        assert!(re.is_laminar());
        assert!(!re.is_turbulent());
    }

    #[test]
    fn test_reynolds_is_turbulent() {
        let re = ReynoldsNumber::new(10.0, 1.0, 0.001);
        assert!(re.is_turbulent());
        assert!(!re.is_laminar());
    }

    #[test]
    fn test_flow_regime_display() {
        assert_eq!(format!("{}", FlowRegime::Laminar), "Laminar");
        assert_eq!(format!("{}", FlowRegime::Turbulent), "Turbulent");
    }

    #[test]
    fn test_flow_regime_description() {
        assert!(!FlowRegime::Creeping.description().is_empty());
        assert!(!FlowRegime::Turbulent.description().is_empty());
    }

    #[test]
    fn test_flow_regime_ranges() {
        let (lo, hi) = FlowRegime::Laminar.re_range();
        assert!((lo - 1.0).abs() < 1e-10);
        assert!((hi - 2300.0).abs() < 1e-10);
    }

    #[test]
    fn test_friction_factor_laminar() {
        let re = ReynoldsNumber::new(0.01, 1.0, 0.01);
        let f = re.friction_factor_pipe(0.0);
        assert!((f - 64.0).abs() < 1.0);
    }

    // === LID-DRIVEN CAVITY ===

    #[test]
    fn test_cavity_creation() {
        let cavity = LidDrivenCavity::new(20, 100.0, 1.0);
        assert_eq!(cavity.solver.nx, 20);
        assert_eq!(cavity.solver.ny, 20);
        assert!((cavity.lid_velocity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cavity_stable_dt() {
        let cavity = LidDrivenCavity::new(20, 100.0, 1.0);
        let dt = cavity.stable_dt(0.3);
        assert!(dt > 0.0);
    }

    #[test]
    fn test_cavity_boundary_conditions() {
        let mut cavity = LidDrivenCavity::new(10, 100.0, 1.0);
        cavity.apply_boundary_conditions();
        // Top wall should have u = lid_velocity
        let top_mid = (9) * 10 + 5;
        assert!((cavity.solver.u[top_mid] - 1.0).abs() < 1e-10);
        // Bottom wall should be zero
        let bot_mid = 5;
        assert!((cavity.solver.u[bot_mid]).abs() < 1e-10);
    }

    #[test]
    fn test_cavity_step() {
        let mut cavity = LidDrivenCavity::new(10, 100.0, 1.0);
        cavity.apply_boundary_conditions();
        let dt = cavity.stable_dt(0.2);
        cavity.step(dt, 20);
        // After some steps, there should be some flow
        assert!(cavity.solver.time > 0.0);
    }

    #[test]
    fn test_cavity_centerline() {
        let mut cavity = LidDrivenCavity::new(10, 100.0, 1.0);
        cavity.apply_boundary_conditions();
        let u_profile = cavity.vertical_centerline_u();
        assert_eq!(u_profile.len(), 10);
        // Top should be lid velocity
        assert!((u_profile[9] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cavity_velocity_magnitude() {
        let mut cavity = LidDrivenCavity::new(10, 100.0, 1.0);
        cavity.apply_boundary_conditions();
        let mag = cavity.velocity_magnitude();
        assert_eq!(mag.len(), 100);
    }

    // === AGENT FLOW ===

    #[test]
    fn test_agent_flow_creation() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let model = AgentFlowModel::new(10, params);
        assert_eq!(model.flow.nx, 10);
    }

    #[test]
    fn test_agent_flow_regime() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 0.1,
            domain_size: 10.0,
            interaction_strength: 0.5,
        };
        let model = AgentFlowModel::new(10, params);
        // Re = 0.1 * 10 / 0.5 = 2.0 → Laminar
        assert_eq!(model.behavior_regime(), FlowRegime::Laminar);
    }

    #[test]
    fn test_agent_flow_uniform_init() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let mut model = AgentFlowModel::new(10, params);
        model.init_uniform(1.0, 0.5, 0.0);
        assert!((model.average_density() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_agent_flow_clustered_init() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let mut model = AgentFlowModel::new(20, params);
        model.init_clustered(5.0, 5.0, 1.0);
        // Center should have higher density
        let center_idx = 10 * 20 + 10;
        let corner_idx = 0;
        assert!(model.property.phi[center_idx] > model.property.phi[corner_idx]);
    }

    #[test]
    fn test_agent_flow_density_variance() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let mut model = AgentFlowModel::new(10, params);
        model.init_uniform(1.0, 0.0, 0.0);
        // Uniform density should have zero variance
        assert!(model.density_variance() < 0.01);
    }

    #[test]
    fn test_agent_flow_detect_clusters() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let mut model = AgentFlowModel::new(20, params);
        model.init_clustered(5.0, 5.0, 0.5);
        let clusters = model.detect_clusters(50.0);
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_agent_flow_summary() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let mut model = AgentFlowModel::new(10, params);
        model.init_uniform(1.0, 0.5, 0.0);
        let summary = model.summary();
        assert!(!summary.regime.is_empty());
        assert!(summary.total_mass > 0.0);
    }

    #[test]
    fn test_agent_flow_step() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let mut model = AgentFlowModel::new(8, params);
        model.init_uniform(1.0, 0.1, 0.0);
        model.step(0.001, 10);
        // Should have advanced
        assert!(model.flow.time > 0.0);
    }

    #[test]
    fn test_agent_flow_flux() {
        let params = AgentFlowParams {
            agent_count: 100.0,
            agent_speed: 1.0,
            domain_size: 10.0,
            interaction_strength: 0.1,
        };
        let mut model = AgentFlowModel::new(10, params);
        model.init_uniform(1.0, 0.5, 0.0);
        let flux = model.flux_through_column(5);
        // Should be positive (flow moving right)
        assert!(flux > 0.0);
    }
}
