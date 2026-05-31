//! Reynolds number computation and flow regime classification.

use serde::{Serialize, Deserialize};

/// Reynolds number with associated flow regime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReynoldsNumber {
    /// The Reynolds number value
    pub re: f64,
    /// Characteristic velocity
    pub velocity: f64,
    /// Characteristic length
    pub length: f64,
    /// Kinematic viscosity
    pub viscosity: f64,
}

impl ReynoldsNumber {
    /// Compute Reynolds number from given parameters.
    pub fn new(velocity: f64, length: f64, viscosity: f64) -> Self {
        Self {
            re: velocity * length / viscosity,
            velocity,
            length,
            viscosity,
        }
    }

    /// Classify the flow regime based on Reynolds number.
    pub fn regime(&self) -> FlowRegime {
        if self.re < 1.0 {
            FlowRegime::Creeping
        } else if self.re < 2300.0 {
            FlowRegime::Laminar
        } else if self.re < 4000.0 {
            FlowRegime::Transitional
        } else {
            FlowRegime::Turbulent
        }
    }

    /// Check if flow is laminar.
    pub fn is_laminar(&self) -> bool {
        self.re < 2300.0
    }

    /// Check if flow is turbulent.
    pub fn is_turbulent(&self) -> bool {
        self.re >= 4000.0
    }

    /// Estimate friction factor using Moody correlation (for pipe flow).
    pub fn friction_factor_pipe(&self, relative_roughness: f64) -> f64 {
        if self.re < 1.0 {
            return 64.0; // Stokes flow limit
        }
        if self.re < 2300.0 {
            // Laminar: f = 64/Re
            return 64.0 / self.re;
        }
        // Turbulent: Swamee-Jain approximation of Colebrook-White
        let e_d = relative_roughness;
        let term = e_d / 3.7 + 5.74 / self.re.powf(0.9);
        let log_term = term.abs().log10();
        0.25 / (log_term * log_term)
    }
}

/// Flow regime classification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FlowRegime {
    /// Re < 1: Stokes/creeping flow (viscosity-dominated)
    Creeping,
    /// 1 ≤ Re < 2300: Laminar (smooth, orderly)
    Laminar,
    /// 2300 ≤ Re < 4000: Transitional (intermittent)
    Transitional,
    /// Re ≥ 4000: Turbulent (chaotic, mixing)
    Turbulent,
}

impl std::fmt::Display for FlowRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlowRegime::Creeping => write!(f, "Creeping (Stokes)"),
            FlowRegime::Laminar => write!(f, "Laminar"),
            FlowRegime::Transitional => write!(f, "Transitional"),
            FlowRegime::Turbulent => write!(f, "Turbulent"),
        }
    }
}

impl FlowRegime {
    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            FlowRegime::Creeping => "Viscosity-dominated flow where inertial forces are negligible",
            FlowRegime::Laminar => "Smooth, orderly flow in parallel layers with no mixing between layers",
            FlowRegime::Transitional => "Flow alternates between laminar and turbulent states",
            FlowRegime::Turbulent => "Chaotic flow with strong mixing, vortices, and energy dissipation",
        }
    }

    /// Typical Reynolds number ranges for each regime.
    pub fn re_range(&self) -> (f64, f64) {
        match self {
            FlowRegime::Creeping => (0.0, 1.0),
            FlowRegime::Laminar => (1.0, 2300.0),
            FlowRegime::Transitional => (2300.0, 4000.0),
            FlowRegime::Turbulent => (4000.0, f64::INFINITY),
        }
    }
}
