use super::math::Vec2D;
use super::{
    components::{Label, Segment},
    style::{Anchor, Decoration, PathStyle, Stroke, Theme},
};
use model::{ParticleBase, ParticleDraw};

pub(crate) mod svg;
pub(crate) mod tikz;
pub(crate) mod typst;

pub trait Backend {
    const BASE_SIZE: f64;
    const BASE_SIZE_X: f64;
    const BASE_SIZE_Y: f64;
    type Output;

    fn particle_name<P: ParticleBase + ParticleDraw>(p: &P) -> &str;

    fn init(size_x: f64, size_y: f64) -> Self;
    fn finish(self) -> Self::Output;

    fn draw_label(&mut self, label: Label);
    fn draw_segment(&mut self, start: Vec2D, segment: &Segment, style: PathStyle);
    fn draw_path(&mut self, start: Vec2D, curve: &Segment, segments: impl Iterator<Item = Segment>, style: PathStyle);
    fn draw_decoration(&mut self, pos: Vec2D, decoration: Decoration);
}
