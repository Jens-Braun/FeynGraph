use std::fmt::Display;

use expression::{Expression, ExpressionAtom};

use model::VertexBase;
use util::{
    HashMap,
    unicode::{index_sub, index_super, indices_sub, indices_super},
};

pub type ColorExpression = Expression<ColorAtom>;

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorAtom {
    Trivial,
    Identity(i64, i64),
    T(i64, i64, i64),
    f(i64, i64, i64),
    d(i64, i64, i64),
    Epsilon(i64, i64, i64),
    EpsilonBar(i64, i64, i64),
    T6(i64, i64, i64),
    K6(i64, i64, i64),
    K6Bar(i64, i64, i64),
}
impl ExpressionAtom for ColorAtom {}

impl Display for ColorAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trivial => write!(f, "𝟙"),
            Self::Identity(a, b) => write!(f, "δ{}", indices_super(&[*a, *b])),
            Self::T(a, i, j) => write!(f, "T{}{}", index_super(*a), indices_sub(&[*i, *j])),
            Self::f(a, b, c) => write!(f, "f{}", indices_super(&[*a, *b, *c])),
            Self::d(a, b, c) => write!(f, "d{}", indices_super(&[*a, *b, *c])),
            Self::Epsilon(i, j, k) => write!(f, "ϵ{}", indices_sub(&[*i, *j, *k])),
            Self::EpsilonBar(i, j, k) => write!(f, "ϵ\u{0305}{}", indices_super(&[*i, *j, *k])),
            Self::T6(a, i, j) => write!(f, "T₆{}{}", index_super(*a), indices_sub(&[*i, *j])),
            Self::K6(a, i, j) => write!(f, "K₆{}{}", indices_super(&[*i, *j]), index_sub(*a)),
            Self::K6Bar(a, i, j) => write!(f, "K\u{0305}₆{}{}", index_super(*a), indices_sub(&[*i, *j])),
        }
    }
}

// impl ReplaceAdjointIndex for ColorAtom {
//     fn replace_adjoint_index(&mut self, query: i64, replace: i64) {
//         let match_replace = |v, q, r| if v == q { r } else { v };
//         match self {
//             Self::Trivial
//             | Self::Epsilon(..)
//             | Self::EpsilonBar(..)
//             | Self::T6(..)
//             | Self::K6(..)
//             | Self::K6Bar(..) => (),
//             Self::Identity(a, b) => {
//                 *a = match_replace(*a, query, replace);
//                 *b = match_replace(*b, query, replace);
//             }
//             Self::T(a, ..) => {
//                 *a = match_replace(*a, query, replace);
//             }
//             Self::f(a, b, c) | Self::d(a, b, c) => {
//                 *a = match_replace(*a, query, replace);
//                 *b = match_replace(*b, query, replace);
//                 *c = match_replace(*c, query, replace);
//             }
//         }
//     }
// }

// impl ReplaceFundamentalIndex for ColorAtom {
//     fn replace_fundamental_index(&mut self, query: i64, replace: i64) {
//         let match_replace = |v, q, r| if v == q { r } else { v };
//         match self {
//             Self::Trivial
//             | Self::Identity(..)
//             | Self::f(..)
//             | Self::d(..)
//             | Self::T6(..)
//             | Self::K6(..)
//             | Self::K6Bar(..) => (),
//             Self::T(_, i, j) => {
//                 *i = match_replace(*i, query, replace);
//                 *j = match_replace(*j, query, replace);
//             }
//             Self::Epsilon(i, j, k) | Self::EpsilonBar(i, j, k) => {
//                 *i = match_replace(*i, query, replace);
//                 *j = match_replace(*j, query, replace);
//                 *k = match_replace(*k, query, replace);
//             }
//         }
//     }
// }

#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vertex {
    pub name: String,
    pub particles: Vec<String>,
    pub color: Vec<Expression<ColorAtom>>,
    pub lorentz: Vec<String>,
    #[cfg_attr(
        feature = "serde",
        serde(serialize_with = "crate::serde_util::serialize_coupling_map")
    )]
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "crate::serde_util::deserialize_coupling_map")
    )]
    pub couplings: HashMap<(usize, usize), String>,

    pub coupling_orders: HashMap<String, usize>,
    pub spin_map: Vec<i64>,
}

impl VertexBase for Vertex {
    fn name(&self) -> &str {
        &self.name
    }

    fn particles(&self) -> &[impl AsRef<str>] {
        &self.particles
    }

    fn coupling_orders(&self) -> &util::HashMap<String, usize> {
        &self.coupling_orders
    }

    fn fermi_map(&self, in_ray: usize) -> usize {
        self.spin_map[in_ray] as usize
    }
}
