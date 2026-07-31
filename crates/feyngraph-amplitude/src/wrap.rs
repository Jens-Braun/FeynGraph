use crate::{AmplitudeAtom, AmpltiudeExpression};
use expression::Expression;
use ufomodel::components::{ColorExpression, LorentzExpression, ParameterExpression};

pub fn wrap_lorentz(expr: LorentzExpression) -> AmpltiudeExpression {
    match expr {
        Expression::IConst(i) => Expression::IConst(i),
        Expression::FConst(f) => Expression::FConst(f),
        Expression::Atom(a) => Expression::Atom(AmplitudeAtom::Lorentz(a)),
        Expression::Inverse(expr) => Expression::Inverse(Box::new(wrap_lorentz(*expr))),
        Expression::Power(x, e) => Expression::Power(Box::new(wrap_lorentz(*x)), Box::new(wrap_lorentz(*e))),
        Expression::Function { module, name, exprs } => Expression::Function {
            module,
            name,
            exprs: exprs.into_iter().map(|e| wrap_lorentz(e)).collect(),
        },
        Expression::Sum(exprs) => Expression::Sum(exprs.into_iter().map(|e| wrap_lorentz(e)).collect()),
        Expression::Product(exprs) => Expression::Product(exprs.into_iter().map(|e| wrap_lorentz(e)).collect()),
    }
}

pub fn wrap_color(expr: ColorExpression) -> AmpltiudeExpression {
    match expr {
        Expression::IConst(i) => Expression::IConst(i),
        Expression::FConst(f) => Expression::FConst(f),
        Expression::Atom(a) => Expression::Atom(AmplitudeAtom::Color(a)),
        Expression::Inverse(expr) => Expression::Inverse(Box::new(wrap_color(*expr))),
        Expression::Power(x, e) => Expression::Power(Box::new(wrap_color(*x)), Box::new(wrap_color(*e))),
        Expression::Function { module, name, exprs } => Expression::Function {
            module,
            name,
            exprs: exprs.into_iter().map(|e| wrap_color(e)).collect(),
        },
        Expression::Sum(exprs) => Expression::Sum(exprs.into_iter().map(|e| wrap_color(e)).collect()),
        Expression::Product(exprs) => Expression::Product(exprs.into_iter().map(|e| wrap_color(e)).collect()),
    }
}

pub fn wrap_parameter(expr: ParameterExpression) -> AmpltiudeExpression {
    match expr {
        Expression::IConst(i) => Expression::IConst(i),
        Expression::FConst(f) => Expression::FConst(f),
        Expression::Atom(a) => Expression::Atom(AmplitudeAtom::Parameter(a)),
        Expression::Inverse(expr) => Expression::Inverse(Box::new(wrap_parameter(*expr))),
        Expression::Power(x, e) => Expression::Power(Box::new(wrap_parameter(*x)), Box::new(wrap_parameter(*e))),
        Expression::Function { module, name, exprs } => Expression::Function {
            module,
            name,
            exprs: exprs.into_iter().map(|e| wrap_parameter(e)).collect(),
        },
        Expression::Sum(exprs) => Expression::Sum(exprs.into_iter().map(|e| wrap_parameter(e)).collect()),
        Expression::Product(exprs) => Expression::Product(exprs.into_iter().map(|e| wrap_parameter(e)).collect()),
    }
}
