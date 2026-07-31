/// Line style of a propagator, specified by the UFO 2.0 standard.
///
/// This property is used for drawing propagators.
#[derive(PartialEq, Debug, Hash, Clone, Eq, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
impl From<&str> for LineStyle {
    fn from(value: &str) -> Self {
        return match value.to_lowercase().as_str() {
            "dashed" => Self::Dashed,
            "dotted" => Self::Dotted,
            "straight" => Self::Straight,
            "wavy" => Self::Wavy,
            "curly" => Self::Curly,
            "scurly" => Self::Scurly,
            "swavy" => Self::Swavy,
            "double" => Self::Double,
            _ => Self::None,
        };
    }
}

pub trait ParticleBase {
    fn name(&self) -> &str;
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
