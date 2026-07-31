use crate::{error::ParseError, parser::diagnostic::UncontractedSpinIndex};
use expression::{Expression, ExpressionAtom /* , ReplaceLorentzIndex, ReplaceSpinIndex*/};
use std::{fmt::Display, panic};
use util::{
    parser::Diagnostic,
    unicode::{digit_super, index_sub, index_super, indices_sub, indices_super},
};

pub type LorentzExpression = Expression<LorentzAtom>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LorentzAtom {
    Trivial,
    Identity(i64, i64),
    IdentityL(i64, i64),
    Gamma(i64, i64, i64),
    Gamma5(i64, i64),
    ProjM(i64, i64),
    ProjP(i64, i64),
    Sigma(i64, i64, i64, i64),
    C(i64, i64),
    Metric(i64, i64),
    P(i64, i64),
    Psq(i64, i64),
    Epsilon(i64, i64, i64, i64),
}
impl ExpressionAtom for LorentzAtom {}

impl Display for LorentzAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trivial => write!(f, "𝟙"),
            Self::Identity(i, j) => write!(f, "δ{}", indices_sub(&[*i, *j])),
            Self::IdentityL(i, j) => write!(f, "δ{}", indices_super(&[*i, *j])),
            Self::Gamma(mu, i, j) => write!(f, "ɣ{}{}", indices_super(&[*mu]), indices_sub(&[*i, *j])),
            Self::Gamma5(i, j) => write!(f, "ɣ{}{}", digit_super(&'5'), indices_sub(&[*i, *j])),
            Self::ProjM(i, j) => write!(f, "Pᴸ{}", indices_sub(&[*i, *j])),
            Self::ProjP(i, j) => write!(f, "Pᴿ{}", indices_sub(&[*i, *j])),
            Self::Sigma(mu, nu, i, j) => write!(f, "σ{}{}", indices_super(&[*mu, *nu]), indices_sub(&[*i, *j])),
            Self::C(i, j) => write!(f, "C{}", indices_sub(&[*i, *j])),
            Self::Metric(i, j) => write!(f, "η{}", indices_super(&[*i, *j])),
            Self::P(i, j) => write!(f, "p{}{}", index_super(*i), index_sub(*j)),
            Self::Psq(_, j) => write!(f, "p²{}", index_sub(*j)),
            Self::Epsilon(mu, nu, rho, sigma) => write!(f, "ϵ{}", indices_super(&[*mu, *nu, *rho, *sigma])),
        }
    }
}

// impl ReplaceLorentzIndex for LorentzAtom {
//     fn replace_lorentz_index(&mut self, query: i64, replace: i64) {
//         let match_replace = |v, q, r| if v == q { r } else { v };
//         match self {
//             Self::Trivial | Self::Identity(..) | Self::Gamma5(..) | Self::ProjM(..) | Self::ProjP(..) | Self::C(..) => {
//                 ()
//             }
//             Self::Gamma(mu, ..) | Self::P(mu, ..) | Self::Psq(mu, ..) => {
//                 *mu = match_replace(*mu, query, replace);
//             }
//             Self::IdentityL(mu, nu) | Self::Sigma(mu, nu, ..) | Self::Metric(mu, nu) => {
//                 *mu = match_replace(*mu, query, replace);
//                 *nu = match_replace(*nu, query, replace);
//             }
//             Self::Epsilon(mu, nu, rho, sigma) => {
//                 *mu = match_replace(*mu, query, replace);
//                 *nu = match_replace(*nu, query, replace);
//                 *rho = match_replace(*rho, query, replace);
//                 *sigma = match_replace(*sigma, query, replace);
//             }
//         }
//     }
// }

// impl ReplaceSpinIndex for LorentzAtom {
//     fn replace_spin_index(&mut self, query: i64, replace: i64) {
//         let match_replace = |v, q, r| if v == q { r } else { v };
//         match self {
//             Self::Trivial
//             | Self::IdentityL(..)
//             | Self::Metric(..)
//             | Self::P(..)
//             | Self::Psq(..)
//             | Self::Epsilon(..) => (),
//             Self::Identity(s1, s2)
//             | Self::Gamma(_, s1, s2)
//             | Self::Gamma5(s1, s2)
//             | Self::ProjM(s1, s2)
//             | Self::ProjP(s1, s2)
//             | Self::Sigma(_, _, s1, s2)
//             | Self::C(s1, s2) => {
//                 *s1 = match_replace(*s1, query, replace);
//                 *s2 = match_replace(*s2, query, replace);
//             }
//         }
//     }
// }

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lorentz {
    pub name: String,
    pub spins: Vec<i64>,
    pub structure: Expression<LorentzAtom>,
    pub rank: i64,
    pub spin_map: Vec<i64>,
}

pub trait Rank {
    fn rank(&self) -> Option<i64>;
    fn ranks(&self) -> Vec<i64>;
}

impl Rank for LorentzExpression {
    fn rank(&self) -> Option<i64> {
        return match self {
            Self::FConst(_) | Self::IConst(_) => Some(0),
            Self::Atom(a) => Some(match a {
                LorentzAtom::P(..) => 1,
                LorentzAtom::Psq(..) => 2,
                _ => 0,
            }),
            Self::Product(exprs) => exprs.iter().map(|e| e.rank()).sum::<Option<i64>>(),
            Expression::Sum(exprs) => {
                let ranks = exprs.iter().map(|e| e.rank()).collect::<Option<Vec<i64>>>();
                if let Some(mut ranks) = ranks {
                    ranks.sort_unstable();
                    ranks.dedup();
                    if ranks.len() != 1 { None } else { ranks.pop() }
                } else {
                    None
                }
            }
            Self::Inverse(expr) => expr.rank().map(|r| -r),
            Self::Function { .. } | Self::Power(..) => unreachable!(),
        };
    }

    fn ranks(&self) -> Vec<i64> {
        return match self {
            Self::Product(exprs) => vec![
                exprs
                    .iter()
                    .map(|e| e.rank())
                    .sum::<Option<i64>>()
                    .expect("Expression is not normalized"),
            ],
            Expression::Sum(exprs) => exprs
                .iter()
                .map(|e| e.rank())
                .collect::<Option<Vec<i64>>>()
                .expect("Expression is not normalized"),
            x @ (Expression::IConst(_) | Expression::FConst(_) | Expression::Atom(_)) => {
                vec![x.rank().expect("Expression is not normalized")]
            }
            _ => panic!("Expression is not normalized"),
        };
    }
}

pub trait SpinMap {
    fn spin_pairs(&self) -> Result<Vec<Vec<(i64, i64)>>, ParseError>;
    fn spin_map(&self) -> Result<Option<Vec<i64>>, ParseError>;
    fn spin_maps(&self) -> Result<Vec<Vec<i64>>, ParseError>;
}

impl SpinMap for LorentzExpression {
    fn spin_pairs(&self) -> Result<Vec<Vec<(i64, i64)>>, ParseError> {
        match self {
            Self::IConst(..) | Self::FConst(..) => return Ok(vec![Vec::new()]),
            Self::Atom(a) => match a {
                LorentzAtom::Trivial
                | LorentzAtom::IdentityL(..)
                | LorentzAtom::Metric(..)
                | LorentzAtom::P(..)
                | LorentzAtom::Psq(..)
                | LorentzAtom::Epsilon(..) => return Ok(vec![Vec::new()]),
                LorentzAtom::Identity(i, j)
                | LorentzAtom::Gamma(_, i, j)
                | LorentzAtom::Gamma5(i, j)
                | LorentzAtom::ProjM(i, j)
                | LorentzAtom::ProjP(i, j)
                | LorentzAtom::Sigma(.., i, j)
                | LorentzAtom::C(i, j) => return Ok(vec![vec![(*i, *j)]]),
            },
            Self::Inverse(expr) => return expr.spin_pairs(),
            Self::Product(exprs) => {
                let mut pairs = {
                    let pairs = exprs
                        .iter()
                        .map(|e| e.spin_pairs())
                        .collect::<Result<Vec<Vec<_>>, _>>()?;
                    if pairs.iter().any(|v| v.len() > 1) {
                        panic!("Expression is not normalized");
                    }
                    pairs.into_iter().flatten().flatten().collect::<Vec<_>>()
                };

                let mut contracted = Vec::new();
                while let Some((i, j)) = pairs.pop() {
                    if i > 0 && j > 0 {
                        contracted.push((i, j));
                    } else {
                        let contract = if i < 0 { i } else { j };
                        let fin = if i < 0 { j } else { i };
                        if let Some(l) = pairs.iter().position(|(i2, j2)| *i2 == contract || *j2 == contract) {
                            let (i2, j2) = pairs.swap_remove(l);
                            if i2 == contract {
                                if j2 < 0 || fin < 0 {
                                    pairs.push((fin, j2));
                                } else {
                                    contracted.push((fin, j2));
                                }
                            } else if j2 == contract {
                                if i2 < 0 || fin < 0 {
                                    pairs.push((i2, fin));
                                } else {
                                    contracted.push((i2, fin));
                                }
                            }
                        } else {
                            let s = self.to_string();
                            dbg!(contract);
                            let span = UncontractedSpinIndex::locate_span(&s, contract);
                            return Err(ParseError::new(
                                "tmp",
                                s,
                                vec![Box::new(UncontractedSpinIndex::new(contract, span)) as Box<dyn Diagnostic>],
                            ));
                        }
                    }
                }
                return Ok(vec![contracted]);
            }
            Self::Sum(exprs) => Ok(exprs
                .iter()
                .map(|e| {
                    let pairs = e.spin_pairs();
                    pairs.map(|mut v| {
                        if v.len() > 1 {
                            panic!("Expression is not normalized");
                        } else {
                            v.pop().unwrap()
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?),
            Self::Function { .. } | Self::Power(..) => unreachable!(),
        }
    }

    fn spin_maps(&self) -> Result<Vec<Vec<i64>>, ParseError> {
        let expr_pairs = self.spin_pairs()?;
        expr_pairs
            .into_iter()
            .map(|pairs| {
                let mut res = vec![0; pairs.iter().map(|(i, j)| (*i).max(*j)).max().unwrap_or(0) as usize];
                for (i, j) in pairs {
                    if i <= 0 || j <= 0 {
                        let contract = if i <= 0 { i } else { j };
                        let s = self.to_string();
                        let span = UncontractedSpinIndex::locate_span(&s, contract);
                        return Err(ParseError::new(
                            "tmp",
                            s,
                            vec![Box::new(UncontractedSpinIndex::new(contract, span)) as Box<dyn Diagnostic>],
                        ));
                    }
                    res[(i - 1) as usize] = j - 1;
                    res[(j - 1) as usize] = i - 1;
                }
                Ok(res)
            })
            .collect()
    }

    fn spin_map(&self) -> Result<Option<Vec<i64>>, ParseError> {
        let mut spin_maps = self.spin_maps()?;
        spin_maps.sort_unstable();
        spin_maps.dedup();
        if spin_maps.len() > 1 {
            return Ok(None);
        } else {
            return Ok(spin_maps.pop());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::components::{
        LorentzAtom,
        lorentz::{LorentzExpression, Rank},
    };
    use expression::Expression;

    #[test]
    fn rank_test() {
        let expr = Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::Gamma5(2, -1));
        assert_eq!(expr.rank(), Some(2));
        let expr = Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::Gamma5(2, -1))
            + Expression::Atom(LorentzAtom::Psq(1, 1)) * Expression::Atom(LorentzAtom::Gamma5(2, -1));
        assert_eq!(expr.rank(), Some(2));
        let expr = Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::Gamma5(2, -1))
            + Expression::Atom(LorentzAtom::Psq(1, 1))
                * Expression::Atom(LorentzAtom::P(1, 1))
                * Expression::Atom(LorentzAtom::Gamma5(2, -1));
        assert_eq!(expr.rank(), None);
        assert_eq!(expr.ranks(), vec![2, 3]);
    }

    #[test]
    fn split_rank_test() {
        let expr = (Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::Gamma5(2, -1))
            + Expression::Atom(LorentzAtom::Psq(1, 1))
                * Expression::Atom(LorentzAtom::P(1, 1))
                * Expression::Atom(LorentzAtom::Gamma5(2, -1)))
        .normalize();
        let exprs = expr.split_by(LorentzExpression::ranks).into_expressions();
        assert_eq!(
            exprs[0],
            (Expression::Atom(LorentzAtom::P(1, 1))
                * Expression::Atom(LorentzAtom::P(1, 1))
                * Expression::Atom(LorentzAtom::Gamma5(2, -1)))
            .normalize(),
        );
        assert_eq!(
            exprs[1],
            (Expression::Atom(LorentzAtom::Psq(1, 1))
                * Expression::Atom(LorentzAtom::P(1, 1))
                * Expression::Atom(LorentzAtom::Gamma5(2, -1)))
            .normalize(),
        );
    }

    #[test]
    fn display_lorentz() {
        let expr = (Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::P(1, 1))
            * Expression::Atom(LorentzAtom::Gamma5(2, -1))
            + Expression::Atom(LorentzAtom::Psq(1, 1))
                * Expression::Atom(LorentzAtom::P(1, 1))
                * Expression::Atom(LorentzAtom::Gamma5(2, -1)))
        .normalize();
        println!("{}", expr);
    }
}
