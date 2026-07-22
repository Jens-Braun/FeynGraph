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

pub trait ParticleBase {
    fn name(&self) -> &str;
    fn id(&self) -> isize;
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
    fn color(&self) -> isize;
}
