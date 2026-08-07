use expression::Expression;
use feyngraph_core::diagram::view::{DiagramView, PropagatorView};
use model::{
    VertexBase,
    analytic::{AmplitudeAtom, AnalyticModel, AnalyticParticle, AnalyticVertex, Direction},
};

mod sort_chains;

pub trait ToAmplitude<M: AnalyticModel> {
    fn to_amplitude(&self) -> Amplitude<M>;
}

pub struct Amplitude<M: AnalyticModel> {
    pub numerator: Expression<M::Atom>,
    pub denominators: Vec<InvProp<M>>,
}

pub struct InvProp<M: AnalyticModel> {
    pub momentum: Expression<M::Atom>,
    pub mass: Expression<M::Atom>,
}

impl<'d, M: AnalyticModel> ToAmplitude<M> for DiagramView<'d, M> {
    fn to_amplitude(&self) -> Amplitude<M> {
        let mut internal_offset: i64 = -1;
        let mut numerator = Expression::IConst(self.sign() as i64);
        for l in self.incoming() {
            numerator *= l
                .particle()
                .external_spinor(l.id(), Direction::Incoming, &mut internal_offset);
        }
        for l in self.outgoing() {
            numerator *= l
                .particle()
                .external_spinor(l.id(), Direction::Outgoing, &mut internal_offset);
        }
        let offset_map = self
            .vertices()
            .scan(0, |acc, v| {
                let current = *acc;
                *acc += v.interaction().degree();
                Some(current)
            })
            .collect::<Vec<_>>();
        for v in self.vertices() {
            let mut rays = (offset_map[v.id()]..(offset_map[v.id()] + v.interaction().degree())).collect::<Vec<_>>();
            v.propagators_ordered().for_each(|p| {
                if let Some(l) = p.left() {
                    rays[l.ray_index_ordered()] = l.id();
                }
            });
            numerator *= v.interaction().structure(&rays, &mut internal_offset);
        }
        for (i, p) in self.propagators().enumerate() {
            let rays = [
                offset_map[p.vertex(0).id()] + p.ray_index_ordered(0),
                offset_map[p.vertex(1).id()] + p.ray_index_ordered(1),
            ];
            numerator *= p
                .particle()
                .propagator_numerator(rays, p.momentum_expr(i as i64), &mut internal_offset)
        }
        let denominators = self
            .propagators()
            .map(|p| {
                let p = InvProp {
                    momentum: p.momentum_expr(internal_offset),
                    mass: p.particle().mass(),
                };
                internal_offset -= 1;
                p
            })
            .collect();
        Amplitude {
            numerator,
            denominators,
        }
    }
}

trait MomentumExpr<M: AnalyticModel> {
    fn momentum_expr(&self, lorentz_index: i64) -> Expression<M::Atom>;
}
impl<M: AnalyticModel> MomentumExpr<M> for PropagatorView<'_, M> {
    fn momentum_expr(&self, lorentz_index: i64) -> Expression<<M as AnalyticModel>::Atom> {
        let mut exprs = self
            .momentum()
            .into_iter()
            .enumerate()
            .filter_map(|(i, x)| match x {
                0 => None,
                1 => Some(M::Atom::momentum(lorentz_index, i as i64)),
                _ => Some(Expression::IConst(x as i64) * M::Atom::momentum(lorentz_index, i as i64)),
            })
            .collect::<Vec<_>>();
        match exprs.len() {
            0 => Expression::IConst(0),
            1 => exprs.pop().unwrap(),
            _ => Expression::Sum(exprs),
        }
    }
}
