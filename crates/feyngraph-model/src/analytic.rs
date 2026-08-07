use itertools::Itertools;

use crate::ModelBase;
use expression::{Expression, ExpressionAtom};

pub enum Direction {
    Incoming,
    Outgoing,
}

pub trait AnalyticModel: ModelBase<Particle: AnalyticParticle<Self::Atom>, Vertex: AnalyticVertex<Self::Atom>> {
    type Atom: ExpressionAtom + AmplitudeAtom;
}

pub trait AnalyticParticle<A: ExpressionAtom> {
    fn propagator_numerator(
        &self,
        global_ray_indices: [usize; 2],
        momentum: Expression<A>,
        internal_offset: &mut i64,
    ) -> Expression<A>;
    fn mass(&self) -> Expression<A>;
    fn external_spinor(&self, leg_index: usize, direction: Direction, internal_offset: &mut i64) -> Expression<A>;
}

pub trait AnalyticVertex<A: ExpressionAtom> {
    fn structure(&self, global_ray_indices: &[usize], internal_offset: &mut i64) -> Expression<A>;
}

pub trait AmplitudeAtom: ExpressionAtom {
    fn momentum(lorentz_index: i64, momentum_index: i64) -> Expression<Self>;
}

pub trait ChainIndices {
    fn chain_indices(&self) -> Option<(i64, i64)>;
}

impl<A: ExpressionAtom + ChainIndices> ChainIndices for Expression<A> {
    fn chain_indices(&self) -> Option<(i64, i64)> {
        match self {
            Self::IConst(_) | Self::FConst(_) | Self::Function { .. } => None,
            Self::Atom(a) => a.chain_indices(),
            Self::Inverse(expr) | Self::Power(expr, _) => expr.chain_indices(),
            Self::Sum(exprs) => {
                let mut pairs = exprs.iter().map(|e| e.chain_indices()).dedup().collect::<Vec<_>>();
                match pairs.len() {
                    0 => None,
                    1 => pairs.pop().unwrap(),
                    x => panic!("Expression contains {x} different chain index pairs in sum"),
                }
            }
            Self::Product(exprs) => {
                let mut pairs = exprs.iter().map(|e| e.chain_indices()).flatten().collect();
                util::resolve_negative(&mut pairs);
                match pairs.len() {
                    0 => None,
                    1 => Some(pairs.pop().unwrap()),
                    x => panic!("Expression contains {x} different chain index pairs in sum"),
                }
            }
        }
    }
}
