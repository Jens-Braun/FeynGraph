use crate::replace::{ReplaceAdjointIndex, ReplaceFundamentalIndex, ReplaceLorentzIndex, ReplaceSpinIndex};
use ufomodel::components::{ColorAtom, LorentzAtom};

use crate::{AmplitudeAtom, AmpltiudeExpression};

#[derive(Debug)]
pub enum Delta {
    Spin(i64, i64),
    Lorentz(i64, i64),
    SUNAdjoint(i64, i64),
    SUNFundamental(i64, i64),
}

impl Delta {
    fn apply(self, expr: &mut AmpltiudeExpression) {
        let (i, j) = match self {
            Self::Lorentz(mu, nu) => (mu, nu),
            Self::Spin(i, j) => (i, j),
            Self::SUNAdjoint(a, b) => (a, b),
            Self::SUNFundamental(i, j) => (i, j),
        };
        let (i, j) = if i > j { (j, i) } else { (i, j) };
        match self {
            Self::Spin(..) => expr.replace_spin_index(i, j),
            Self::Lorentz(..) => expr.replace_lorentz_index(i, j),
            Self::SUNAdjoint(..) => expr.replace_adjoint_index(i, j),
            Self::SUNFundamental(..) => expr.replace_fundamental_index(i, j),
        }
    }

    fn internal(&self) -> bool {
        let (i, j) = match self {
            Self::Lorentz(mu, nu) => (mu, nu),
            Self::Spin(i, j) => (i, j),
            Self::SUNAdjoint(a, b) => (a, b),
            Self::SUNFundamental(i, j) => (i, j),
        };
        *i < 0 && *j < 0
    }
}

pub trait ApplyDelta {
    fn apply_delta(&mut self);
}

impl ApplyDelta for AmpltiudeExpression {
    fn apply_delta(&mut self) {
        match self {
            Self::IConst(_) | Self::FConst(_) | Self::Atom(_) => (),
            Self::Function { exprs, .. } | Self::Sum(exprs) => {
                for expr in exprs.iter_mut() {
                    expr.apply_delta();
                }
            }
            Self::Power(x, e) => {
                x.apply_delta();
                e.apply_delta();
            }
            Self::Inverse(expr) => {
                expr.apply_delta();
            }
            Self::Product(exprs) => {
                let mut indices = Vec::new();
                exprs.retain(|expr| match expr {
                    Self::Atom(a) => {
                        if let Some(delta) = a.delta_indices()
                            && delta.internal()
                        {
                            indices.push(delta);
                            false
                        } else {
                            true
                        }
                    }
                    _ => true,
                });
                for d in indices {
                    d.apply(self);
                }
            }
        }
    }
}

trait DeltaIndices {
    fn delta_indices(&self) -> Option<Delta>;
}

impl DeltaIndices for AmplitudeAtom {
    fn delta_indices(&self) -> Option<Delta> {
        match self {
            Self::Color(a) => a.delta_indices(),
            Self::Lorentz(a) => a.delta_indices(),
            Self::Parameter(_) => None,
            Self::External(_) => None,
        }
    }
}

impl DeltaIndices for LorentzAtom {
    fn delta_indices(&self) -> Option<Delta> {
        match self {
            Self::Identity(i, j) => Some(Delta::Spin(*i, *j)),
            Self::IdentityL(mu, nu) => Some(Delta::Lorentz(*mu, *nu)),
            Self::Metric(mu, nu) => Some(Delta::Lorentz(*mu, *nu)),
            _ => None,
        }
    }
}

impl DeltaIndices for ColorAtom {
    fn delta_indices(&self) -> Option<Delta> {
        match self {
            ColorAtom::Identity(a, b) => {
                if *a < -100 {
                    Some(Delta::SUNAdjoint(*a, *b))
                } else {
                    Some(Delta::SUNFundamental(*a, *b))
                }
            }
            _ => None,
        }
    }
}
