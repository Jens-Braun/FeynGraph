use std::fmt::Display;

use expression::{Expression, ExpressionAtom};
use num_complex::Complex;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterValue {
    ExternalReal(f64),
    // Forbidden by UFO2 standard, but used anyways...
    ExternalComplex(Complex<f64>),
    InternalReal(ParameterExpression),
    InternalComplex(ParameterExpression),
}

pub type ParameterExpression = Expression<ParameterAtom>;

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterAtom {
    I,
    Constant(String),
    Parameter(String),
}

impl Display for ParameterAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I => write!(f, "i"),
            Self::Constant(c) => write!(f, "{}", c.to_uppercase()),
            Self::Parameter(p) => write!(f, "{p}"),
        }
    }
}

impl ExpressionAtom for ParameterAtom {}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Parameter {
    pub(crate) name: String,
    pub(crate) value: ParameterValue,
    pub(crate) texname: String,
    pub(crate) lha_block: Option<String>,
    pub(crate) lha_code: Option<Vec<usize>>,
}
