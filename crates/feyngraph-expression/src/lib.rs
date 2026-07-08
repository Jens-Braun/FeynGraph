mod display;
mod normalize;
mod ops;
mod replace;
mod split;
mod translation;

pub use translation::{Dictionary, Translator};

pub trait ExpressionAtom: PartialEq + PartialOrd + Clone {}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expression<A: ExpressionAtom> {
    IConst(i64),
    FConst(f64),
    Atom(A),
    Sum(Vec<Expression<A>>),
    Product(Vec<Expression<A>>),
    Inverse(Box<Expression<A>>),
    Power(Box<Expression<A>>, Box<Expression<A>>),
    Function {
        module: Option<String>,
        name: String,
        exprs: Vec<Expression<A>>,
    },
}
