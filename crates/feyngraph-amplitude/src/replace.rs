use crate::{Expression, ExpressionAtom};
use ufomodel::components::{ColorAtom, LorentzAtom};

pub trait ReplaceSpinIndex {
    fn replace_spin_index(&mut self, query: i64, replace: i64);
}

impl<A: ExpressionAtom + ReplaceSpinIndex> ReplaceSpinIndex for Expression<A> {
    fn replace_spin_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::IConst(_) | Self::FConst(_) => (),
            Self::Atom(a) => a.replace_spin_index(query, replace),
            Self::Sum(exprs) | Self::Product(exprs) | Self::Function { exprs, .. } => {
                exprs
                    .iter_mut()
                    .for_each(|expr| expr.replace_spin_index(query, replace));
            }
            Self::Inverse(expr) => expr.replace_spin_index(query, replace),
            Self::Power(x, e) => {
                x.replace_spin_index(query, replace);
                e.replace_spin_index(query, replace);
            }
        }
    }
}

pub trait ReplaceLorentzIndex {
    fn replace_lorentz_index(&mut self, query: i64, replace: i64);
}

impl<A: ExpressionAtom + ReplaceLorentzIndex> ReplaceLorentzIndex for Expression<A> {
    fn replace_lorentz_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::IConst(_) | Self::FConst(_) => (),
            Self::Atom(a) => a.replace_lorentz_index(query, replace),
            Self::Sum(exprs) | Self::Product(exprs) | Self::Function { exprs, .. } => {
                exprs
                    .iter_mut()
                    .for_each(|expr| expr.replace_lorentz_index(query, replace));
            }
            Self::Inverse(expr) => expr.replace_lorentz_index(query, replace),
            Self::Power(x, e) => {
                x.replace_lorentz_index(query, replace);
                e.replace_lorentz_index(query, replace);
            }
        }
    }
}

pub trait ReplaceAdjointIndex {
    fn replace_adjoint_index(&mut self, query: i64, replace: i64);
}

impl<A: ExpressionAtom + ReplaceAdjointIndex> ReplaceAdjointIndex for Expression<A> {
    fn replace_adjoint_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::IConst(_) | Self::FConst(_) => (),
            Self::Atom(a) => a.replace_adjoint_index(query, replace),
            Self::Sum(exprs) | Self::Product(exprs) | Self::Function { exprs, .. } => {
                exprs
                    .iter_mut()
                    .for_each(|expr| expr.replace_adjoint_index(query, replace));
            }
            Self::Inverse(expr) => expr.replace_adjoint_index(query, replace),
            Self::Power(x, e) => {
                x.replace_adjoint_index(query, replace);
                e.replace_adjoint_index(query, replace);
            }
        }
    }
}

pub trait ReplaceFundamentalIndex {
    fn replace_fundamental_index(&mut self, query: i64, replace: i64);
}

impl<A: ExpressionAtom + ReplaceFundamentalIndex> ReplaceFundamentalIndex for Expression<A> {
    fn replace_fundamental_index(&mut self, query: i64, replace: i64) {
        match self {
            Self::IConst(_) | Self::FConst(_) => (),
            Self::Atom(a) => a.replace_fundamental_index(query, replace),
            Self::Sum(exprs) | Self::Product(exprs) | Self::Function { exprs, .. } => {
                exprs
                    .iter_mut()
                    .for_each(|expr| expr.replace_fundamental_index(query, replace));
            }
            Self::Inverse(expr) => expr.replace_fundamental_index(query, replace),
            Self::Power(x, e) => {
                x.replace_fundamental_index(query, replace);
                e.replace_fundamental_index(query, replace);
            }
        }
    }
}

impl ReplaceLorentzIndex for LorentzAtom {
    fn replace_lorentz_index(&mut self, query: i64, replace: i64) {
        let match_replace = |v, q, r| if v == q { r } else { v };
        match self {
            Self::Trivial | Self::Identity(..) | Self::Gamma5(..) | Self::ProjM(..) | Self::ProjP(..) | Self::C(..) => {
                ()
            }
            Self::Gamma(mu, ..) | Self::P(mu, ..) | Self::Psq(mu, ..) => {
                *mu = match_replace(*mu, query, replace);
            }
            Self::IdentityL(mu, nu) | Self::Sigma(mu, nu, ..) | Self::Metric(mu, nu) => {
                *mu = match_replace(*mu, query, replace);
                *nu = match_replace(*nu, query, replace);
            }
            Self::Epsilon(mu, nu, rho, sigma) => {
                *mu = match_replace(*mu, query, replace);
                *nu = match_replace(*nu, query, replace);
                *rho = match_replace(*rho, query, replace);
                *sigma = match_replace(*sigma, query, replace);
            }
        }
    }
}

impl ReplaceSpinIndex for LorentzAtom {
    fn replace_spin_index(&mut self, query: i64, replace: i64) {
        let match_replace = |v, q, r| if v == q { r } else { v };
        match self {
            Self::Trivial
            | Self::IdentityL(..)
            | Self::Metric(..)
            | Self::P(..)
            | Self::Psq(..)
            | Self::Epsilon(..) => (),
            Self::Identity(s1, s2)
            | Self::Gamma(_, s1, s2)
            | Self::Gamma5(s1, s2)
            | Self::ProjM(s1, s2)
            | Self::ProjP(s1, s2)
            | Self::Sigma(_, _, s1, s2)
            | Self::C(s1, s2) => {
                *s1 = match_replace(*s1, query, replace);
                *s2 = match_replace(*s2, query, replace);
            }
        }
    }
}

impl ReplaceAdjointIndex for ColorAtom {
    fn replace_adjoint_index(&mut self, query: i64, replace: i64) {
        let match_replace = |v, q, r| if v == q { r } else { v };
        match self {
            Self::Trivial
            | Self::Epsilon(..)
            | Self::EpsilonBar(..)
            | Self::T6(..)
            | Self::K6(..)
            | Self::K6Bar(..) => (),
            Self::Identity(a, b) => {
                *a = match_replace(*a, query, replace);
                *b = match_replace(*b, query, replace);
            }
            Self::T(a, ..) => {
                *a = match_replace(*a, query, replace);
            }
            Self::f(a, b, c) | Self::d(a, b, c) => {
                *a = match_replace(*a, query, replace);
                *b = match_replace(*b, query, replace);
                *c = match_replace(*c, query, replace);
            }
        }
    }
}

impl ReplaceFundamentalIndex for ColorAtom {
    fn replace_fundamental_index(&mut self, query: i64, replace: i64) {
        let match_replace = |v, q, r| if v == q { r } else { v };
        match self {
            Self::Trivial
            | Self::Identity(..)
            | Self::f(..)
            | Self::d(..)
            | Self::T6(..)
            | Self::K6(..)
            | Self::K6Bar(..) => (),
            Self::T(_, i, j) => {
                *i = match_replace(*i, query, replace);
                *j = match_replace(*j, query, replace);
            }
            Self::Epsilon(i, j, k) | Self::EpsilonBar(i, j, k) => {
                *i = match_replace(*i, query, replace);
                *j = match_replace(*j, query, replace);
                *k = match_replace(*k, query, replace);
            }
        }
    }
}
