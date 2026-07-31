use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::ParseError;
use expression::{Expression, ExpressionAtom};
use util::parser::Diagnostic;

#[derive(Debug)]
pub enum ExpressionAst<A: ExpressionAtom> {
    Error(Vec<Box<dyn Diagnostic>>),
    Sum(Vec<ExpressionAst<A>>),
    Product(Vec<ExpressionAst<A>>),
    Negative(Box<ExpressionAst<A>>),
    Inverse(Box<ExpressionAst<A>>),
    Power(Box<ExpressionAst<A>>, Box<ExpressionAst<A>>),
    Function {
        module: Option<String>,
        name: String,
        exprs: Vec<ExpressionAst<A>>,
    },
    Atom(A),
    IConst(i64),
    FConst(f64),
}

impl<A: ExpressionAtom> ExpressionAst<A> {
    pub(crate) fn into_result(self, name: &str, input: &str) -> Result<Expression<A>, ParseError> {
        match self {
            Self::Atom(a) => Ok(Expression::Atom(a)),
            Self::IConst(i) => Ok(Expression::IConst(i)),
            Self::FConst(f) => Ok(Expression::FConst(f)),
            Self::Negative(inner) => inner.into_result(name, input).map(|e| -e),
            Self::Inverse(inner) => inner.into_result(name, input).map(|e| Expression::Inverse(Box::new(e))),
            Self::Power(x, e) => match (x.into_result(name, input), e.into_result(name, input)) {
                (Ok(x), Ok(y)) => Ok(Expression::Power(Box::new(x), Box::new(y))),
                (e @ Err(_), _) => return e,
                (_, e @ Err(_)) => return e,
            },
            Self::Function { module, name: f, exprs } => {
                let inner_results: Vec<_> = exprs.into_iter().map(|ast| ast.into_result(name, input)).collect();
                if inner_results.iter().any(|x| x.is_err()) {
                    return Err(ParseError::new(
                        name,
                        input,
                        inner_results
                            .into_iter()
                            .filter_map(|r| match r {
                                Ok(_) => None,
                                Err(e) => Some(e.into_diagnostics()),
                            })
                            .flatten()
                            .collect(),
                    ));
                } else {
                    return Ok(Expression::Function {
                        module,
                        name: f,
                        exprs: inner_results.into_iter().map(|r| r.unwrap()).collect(),
                    });
                }
            }
            Self::Sum(inner) => {
                let inner_results: Vec<_> = inner.into_iter().map(|ast| ast.into_result(name, input)).collect();
                if inner_results.iter().any(|x| x.is_err()) {
                    return Err(ParseError::new(
                        name,
                        input,
                        inner_results
                            .into_iter()
                            .filter_map(|r| match r {
                                Ok(_) => None,
                                Err(e) => Some(e.into_diagnostics()),
                            })
                            .flatten()
                            .collect(),
                    ));
                } else {
                    return Ok(Expression::Sum(inner_results.into_iter().map(|r| r.unwrap()).collect()));
                }
            }
            Self::Product(inner) => {
                let inner_results: Vec<_> = inner.into_iter().map(|ast| ast.into_result(name, input)).collect();
                if inner_results.iter().any(|x| x.is_err()) {
                    return Err(ParseError::new(
                        name,
                        input,
                        inner_results
                            .into_iter()
                            .filter_map(|r| match r {
                                Ok(_) => None,
                                Err(e) => Some(e.into_diagnostics()),
                            })
                            .flatten()
                            .collect(),
                    ));
                } else {
                    return Ok(Expression::Product(
                        inner_results.into_iter().map(|r| r.unwrap()).collect(),
                    ));
                }
            }
            Self::Error(errs) => Err(ParseError::new(name, input, errs)),
        }
    }
}

impl<A: ExpressionAtom> Add<ExpressionAst<A>> for ExpressionAst<A> {
    type Output = Self;

    fn add(self, rhs: ExpressionAst<A>) -> Self::Output {
        return match (self, rhs) {
            (ExpressionAst::Sum(mut x), ExpressionAst::Sum(mut y)) => {
                x.append(&mut y);
                ExpressionAst::Sum(x)
            }
            (ExpressionAst::Sum(mut x), y) => {
                x.push(y);
                ExpressionAst::Sum(x)
            }
            (x, ExpressionAst::Sum(mut y)) => {
                y.push(x);
                ExpressionAst::Sum(y)
            }
            (ExpressionAst::Negative(x), ExpressionAst::Negative(y)) => ExpressionAst::Negative(Box::new(*x + *y)),
            (x, y) => ExpressionAst::Sum(vec![x, y]),
        };
    }
}

impl<A: ExpressionAtom> Sub<ExpressionAst<A>> for ExpressionAst<A> {
    type Output = Self;

    fn sub(self, rhs: ExpressionAst<A>) -> Self::Output {
        self + (-rhs)
    }
}

impl<A: ExpressionAtom> Neg for ExpressionAst<A> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            ExpressionAst::Negative(x) => *x,
            x => ExpressionAst::Negative(Box::new(x)),
        }
    }
}

impl<A: ExpressionAtom> Mul<ExpressionAst<A>> for ExpressionAst<A> {
    type Output = Self;

    fn mul(self, rhs: ExpressionAst<A>) -> Self::Output {
        return match (self, rhs) {
            (ExpressionAst::Product(mut x), ExpressionAst::Product(mut y)) => {
                x.append(&mut y);
                ExpressionAst::Product(x)
            }
            (ExpressionAst::Product(mut x), y) => {
                x.push(y);
                ExpressionAst::Product(x)
            }
            (x, ExpressionAst::Product(mut y)) => {
                y.push(x);
                ExpressionAst::Product(y)
            }
            (ExpressionAst::Negative(x), ExpressionAst::Negative(y)) => *x * *y,
            (x, ExpressionAst::Negative(y)) => ExpressionAst::Negative(Box::new(x * *y)),
            (ExpressionAst::Negative(x), y) => ExpressionAst::Negative(Box::new(*x * y)),
            (ExpressionAst::Inverse(x), ExpressionAst::Inverse(y)) => ExpressionAst::Inverse(Box::new(*x * *y)),
            (x, y) => ExpressionAst::Product(vec![x, y]),
        };
    }
}

impl<A: ExpressionAtom> Div<ExpressionAst<A>> for ExpressionAst<A> {
    type Output = Self;

    fn div(self, rhs: ExpressionAst<A>) -> Self::Output {
        self * ExpressionAst::Inverse(Box::new(rhs))
    }
}
