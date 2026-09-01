use std::fmt::Display;

use log::warn;

/// Line style of a propagator, specified by the UFO 2.0 standard.
///
/// This property is used for drawing propagators.
#[derive(PartialEq, Debug, Hash, Clone, Eq, Copy)]
pub enum LineStyle {
    Dashed,
    Dotted,
    Straight,
    Wavy,
    Curly,
    Scurly,
    Swavy,
    Double,
    None,
}

impl From<String> for LineStyle {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "dashed" => Self::Dashed,
            "dotted" => Self::Dotted,
            "straight" => Self::Straight,
            "wavy" => Self::Wavy,
            "curly" => Self::Curly,
            "scurly" => Self::Scurly,
            "Swavy" => Self::Swavy,
            "double" => Self::Double,
            "none" => Self::None,
            x => {
                warn!("Ignoring unknown linestyle `{x}`");
                LineStyle::None
            }
        }
    }
}

impl Display for LineStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LineStyle::Dashed => write!(f, "dashed"),
            LineStyle::Dotted => write!(f, "dotted"),
            LineStyle::Straight => write!(f, "straight"),
            LineStyle::Wavy => write!(f, "wavy"),
            LineStyle::Curly => write!(f, "curly"),
            LineStyle::Scurly => write!(f, "scurly"),
            LineStyle::Swavy => write!(f, "swavy"),
            LineStyle::Double => write!(f, "double"),
            LineStyle::None => write!(f, "none"),
        }
    }
}

pub trait ParticleBase {
    fn name(&self) -> &str;
    fn anti_name(&self) -> &str;
    fn id(&self) -> i64;
    fn is_self_anti(&self) -> bool;
    fn is_fermi(&self) -> bool;

    fn is_anti(&self) -> bool {
        self.id() < 0
    }
}

pub trait ParticleDraw {
    fn display_name(&self) -> &str;
    fn linestyle(&self) -> LineStyle;
}

pub trait ParticleColor {
    fn color(&self) -> i64;
}
