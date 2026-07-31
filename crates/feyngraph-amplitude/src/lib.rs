use std::fmt::Display;

use expression::{Expression, ExpressionAtom};
use feyngraph_core::diagram::view::*;
use model::{ParticleBase, VertexBase};
use ufomodel::{
    UFOModel,
    components::{ColorAtom, CouplingValue, LorentzAtom, LorentzExpression, ParameterAtom, ParameterExpression},
};
use util::unicode::{index_sub, index_super};

use replace::{ReplaceAdjointIndex, ReplaceFundamentalIndex, ReplaceLorentzIndex, ReplaceSpinIndex};
pub use sort::SortSpinChains;

use crate::apply_delta::ApplyDelta;

mod apply_delta;
mod replace;
mod sort;
pub mod wrap;

#[derive(PartialEq, PartialOrd, Clone, Debug)]
pub enum AmplitudeAtom {
    Parameter(ParameterAtom),
    Color(ColorAtom),
    Lorentz(LorentzAtom),
    External(ExternalAtom),
}

impl Display for AmplitudeAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lorentz(expr) => write!(f, "{}", expr),
            Self::Color(expr) => write!(f, "{}", expr),
            Self::Parameter(expr) => write!(f, "{}", expr),
            Self::External(expr) => write!(f, "{}", expr),
        }
    }
}

impl ExpressionAtom for AmplitudeAtom {}

pub type AmpltiudeExpression = Expression<AmplitudeAtom>;

impl ReplaceLorentzIndex for AmplitudeAtom {
    fn replace_lorentz_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::Lorentz(expr) => expr.replace_lorentz_index(query, replace),
            _ => (),
        }
    }
}

impl ReplaceSpinIndex for AmplitudeAtom {
    fn replace_spin_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::Lorentz(expr) => expr.replace_spin_index(query, replace),
            _ => (),
        }
    }
}

impl ReplaceAdjointIndex for AmplitudeAtom {
    fn replace_adjoint_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::Color(expr) => expr.replace_adjoint_index(query, replace),
            _ => (),
        }
    }
}

impl ReplaceFundamentalIndex for AmplitudeAtom {
    fn replace_fundamental_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::Color(expr) => expr.replace_fundamental_index(query, replace),
            _ => (),
        }
    }
}

#[derive(PartialEq, PartialOrd, Clone, Debug)]
pub enum ExternalAtom {
    U(i64, i64, String),
    Ubar(i64, i64, String),
    V(i64, i64, String),
    Vbar(i64, i64, String),
    Epsilon(i64, i64),
    EpsilonConj(i64, i64),
}

impl Display for ExternalAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(s, i, _) => write!(f, "u{}(p{})", index_sub(*s), index_sub(*i)),
            Self::Ubar(s, i, _) => write!(f, "u\u{0305}{}(p{})", index_sub(*s), index_sub(*i)),
            Self::V(s, i, _) => write!(f, "v{}(p{})", index_sub(*s), index_sub(*i)),
            Self::Vbar(s, i, _) => write!(f, "v\u{0305}{}(p{})", index_sub(*s), index_sub(*i)),
            Self::Epsilon(mu, i) => write!(f, "ε{}(p{})", index_super(*mu), index_sub(*i)),
            Self::EpsilonConj(mu, i) => write!(f, "ε⁺{}(p{})", index_super(*mu), index_sub(*i)),
        }
    }
}

impl ExpressionAtom for ExternalAtom {}

#[derive(Debug, Clone)]
pub struct PropagatorDenominator {
    pub momentum: LorentzExpression,
    pub mass: ParameterExpression,
}

impl Display for PropagatorDenominator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} - {}", self.momentum, self.mass)
    }
}

#[derive(Debug)]
pub struct ScatteringAmplitude {
    pub numerator: Expression<AmplitudeAtom>,
    pub denominators: Vec<PropagatorDenominator>,
}

impl Display for ScatteringAmplitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Scattering Amplitude {{
    numerator: {},
    denominators: {}
}}",
            self.numerator,
            format!(
                "[{}]",
                self.denominators
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        )
    }
}

pub trait ToAmplitude {
    fn to_amplitude(self) -> ScatteringAmplitude;
}

impl<'d> ToAmplitude for DiagramView<'d, UFOModel> {
    fn to_amplitude(self) -> ScatteringAmplitude {
        let mut denoms = Vec::with_capacity(self.propagators().count());
        let mut numerator = vec![Expression::IConst(self.sign() as i64)];
        let mut index = 1;
        for l in self.incoming() {
            numerator.push(leg_in(l.clone(), index));
            index += 1
        }
        for l in self.outgoing() {
            numerator.push(leg_out(l.clone(), index));
            index += 1
        }
        let mut color_offset = -1;
        let mut spin_offset = -1;
        let mut lorentz_offset = -1;
        for v in self.vertices() {
            numerator.push(vertex_expr(v, &mut color_offset, &mut spin_offset, &mut lorentz_offset));
        }

        for p in self.propagators() {
            numerator.push(propagator_numerator(&self, p.clone(), &mut lorentz_offset));
            denoms.push(PropagatorDenominator {
                momentum: momentum_expr(p.clone(), 1),
                mass: Expression::Atom(ParameterAtom::Parameter(p.particle().mass.clone())),
            });
        }

        let mut numerator = Expression::Product(numerator);

        for (i, l) in self.legs().enumerate() {
            let query = vertex_offset(&self, l.vertex()) - l.ray_index_ordered() as i64;
            numerator.replace_lorentz_index(query, i as i64 + 1);
            numerator.replace_spin_index(query, i as i64 + 1);
            match l.particle().color.abs() {
                1 => (),
                3 => numerator.replace_fundamental_index(query, i as i64 + 1),
                8 => numerator.replace_adjoint_index(query - 100, i as i64 + 1 + 100),
                x => unimplemented!("{}", x),
            }
        }

        numerator = numerator.normalize();
        numerator.sort_chains();
        numerator.apply_delta();

        return ScatteringAmplitude {
            numerator,
            denominators: denoms,
        };
    }
}

fn momentum_expr(prop: PropagatorView<UFOModel>, lorentz_index: i64) -> LorentzExpression {
    let mut exprs = prop
        .momentum()
        .into_iter()
        .enumerate()
        .filter_map(|(i, c)| match c {
            0 => None,
            1 => Some(Expression::Atom(LorentzAtom::P(lorentz_index, i as i64 + 1))),
            _ => Some(Expression::Product(vec![
                Expression::IConst(c as i64),
                Expression::Atom(LorentzAtom::P(lorentz_index, i as i64 + 1)),
            ])),
        })
        .collect::<Vec<_>>();
    match exprs.len() {
        0 => Expression::IConst(0),
        1 => exprs.pop().unwrap(),
        _ => Expression::Sum(exprs),
    }
}

fn propagator_numerator(
    diag: &DiagramView<UFOModel>,
    prop: PropagatorView<UFOModel>,
    contract_index: &mut i64,
) -> Expression<AmplitudeAtom> {
    let p = prop.particle();
    match (p.spin.abs(), p.color.abs()) {
        (1, 1) => Expression::Atom(AmplitudeAtom::Parameter(ParameterAtom::I)),
        (1, 8) => Expression::Product(vec![
            Expression::Atom(AmplitudeAtom::Parameter(ParameterAtom::I)),
            Expression::Atom(AmplitudeAtom::Color(ColorAtom::Identity(
                vertex_offset(diag, prop.vertex(0)) - prop.ray_index_ordered(0) as i64,
                vertex_offset(diag, prop.vertex(1)) - prop.ray_index_ordered(1) as i64,
            ))),
        ]),
        (2, _) => {
            let tmp = Expression::Product(vec![
                Expression::Atom(AmplitudeAtom::Parameter(ParameterAtom::I)),
                Expression::Sum(vec![
                    Expression::Product(vec![
                        Expression::Atom(AmplitudeAtom::Lorentz(LorentzAtom::Gamma(
                            *contract_index,
                            vertex_offset(diag, prop.vertex(0)) - prop.ray_index_ordered(0) as i64,
                            vertex_offset(diag, prop.vertex(1)) - prop.ray_index_ordered(1) as i64,
                        ))),
                        wrap::wrap_lorentz(momentum_expr(prop.clone(), *contract_index)),
                    ]),
                    Expression::Product(vec![
                        Expression::Atom(AmplitudeAtom::Lorentz(LorentzAtom::Identity(
                            vertex_offset(diag, prop.vertex(0)) - prop.ray_index_ordered(0) as i64,
                            vertex_offset(diag, prop.vertex(1)) - prop.ray_index_ordered(1) as i64,
                        ))),
                        Expression::Atom(AmplitudeAtom::Parameter(ParameterAtom::Parameter(p.mass.clone()))),
                    ]),
                ]),
            ]);
            *contract_index -= 1;
            tmp
        }
        (3, 1) => Expression::Product(vec![
            Expression::IConst(-1),
            Expression::Atom(AmplitudeAtom::Parameter(ParameterAtom::I)),
            Expression::Atom(AmplitudeAtom::Lorentz(LorentzAtom::Metric(
                vertex_offset(diag, prop.vertex(0)) - prop.ray_index_ordered(0) as i64,
                vertex_offset(diag, prop.vertex(1)) - prop.ray_index_ordered(1) as i64,
            ))),
        ]),
        (3, 8) => Expression::Product(vec![
            Expression::IConst(-1),
            Expression::Atom(AmplitudeAtom::Parameter(ParameterAtom::I)),
            Expression::Atom(AmplitudeAtom::Color(ColorAtom::Identity(
                vertex_offset(diag, prop.vertex(0)) - 100 - prop.ray_index_ordered(0) as i64,
                vertex_offset(diag, prop.vertex(1)) - 100 - prop.ray_index_ordered(1) as i64,
            ))),
            Expression::Atom(AmplitudeAtom::Lorentz(LorentzAtom::Metric(
                vertex_offset(diag, prop.vertex(0)) - prop.ray_index_ordered(0) as i64,
                vertex_offset(diag, prop.vertex(1)) - prop.ray_index_ordered(1) as i64,
            ))),
        ]),
        _ => unimplemented!(),
    }
}

fn vertex_expr(
    vert: VertexView<UFOModel>,
    color_offset: &mut i64,
    spin_offset: &mut i64,
    lorentz_offset: &mut i64,
) -> Expression<AmplitudeAtom> {
    let v = vert.interaction();
    let mut exprs = v
        .couplings
        .iter()
        .map(|((c, l), gc)| {
            Expression::Product(vec![
                match &vert.model().couplings.get(gc).unwrap().value {
                    CouplingValue::Finite(expr) => wrap::wrap_parameter(expr.clone()),
                    _ => unimplemented!(),
                },
                {
                    let mut expr = v.color[*c].clone();
                    for i in 1..=(v.particles.len() as i64) {
                        if vert.model().particles.get(&v.particles[i as usize - 1]).unwrap().color == 8 {
                            expr.replace_adjoint_index(i, *color_offset - i + 1 - 100);
                        } else {
                            expr.replace_fundamental_index(i, *color_offset - i + 1);
                        }
                    }
                    *color_offset -= v.particles.len() as i64;
                    wrap::wrap_color(expr)
                },
                {
                    let l = vert.model().lorentz.get(&v.lorentz[*l]).unwrap();
                    let mut expr = l.structure.clone();
                    for i in 1..=(l.spins.len() as i64) {
                        expr.replace_spin_index(i, *spin_offset - i + 1);
                        expr.replace_lorentz_index(i, *lorentz_offset - i + 1);
                    }
                    *spin_offset -= l.spins.len() as i64;
                    *lorentz_offset -= l.spins.len() as i64;
                    wrap::wrap_lorentz(expr)
                },
            ])
        })
        .collect::<Vec<_>>();
    match exprs.len() {
        0 => Expression::IConst(0),
        1 => exprs.pop().unwrap(),
        _ => Expression::Product(exprs),
    }
}

fn leg_in(leg: LegView<UFOModel>, id: i64) -> Expression<AmplitudeAtom> {
    match leg.particle().spin.abs() + 1 {
        1 => Expression::IConst(1),
        2 => match leg.particle().is_anti() {
            true => Expression::Atom(AmplitudeAtom::External(ExternalAtom::Vbar(
                id,
                id,
                leg.particle().mass.clone(),
            ))),
            false => Expression::Atom(AmplitudeAtom::External(ExternalAtom::U(
                id,
                id,
                leg.particle().mass.clone(),
            ))),
        },
        3 => Expression::Atom(AmplitudeAtom::External(ExternalAtom::Epsilon(id, id))),
        _ => unimplemented!(),
    }
}

fn leg_out(leg: LegView<UFOModel>, id: i64) -> Expression<AmplitudeAtom> {
    match leg.particle().spin.abs() + 1 {
        1 => Expression::IConst(1),
        2 => match leg.particle().is_anti() {
            true => Expression::Atom(AmplitudeAtom::External(ExternalAtom::V(
                id,
                id,
                leg.particle().mass.clone(),
            ))),
            false => Expression::Atom(AmplitudeAtom::External(ExternalAtom::Ubar(
                id,
                id,
                leg.particle().mass.clone(),
            ))),
        },
        3 => Expression::Atom(AmplitudeAtom::External(ExternalAtom::EpsilonConj(id, id))),
        _ => unimplemented!(),
    }
}

fn vertex_offset(diag: &DiagramView<UFOModel>, vert: VertexView<UFOModel>) -> i64 {
    -diag
        .vertices()
        .map_while(|v| {
            if v.id() != vert.id() {
                Some(v.interaction().particles().len() as i64)
            } else {
                None
            }
        })
        .sum::<i64>()
        - 1
}
