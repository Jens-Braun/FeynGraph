use std::fmt::{Display, Error, Write};

use feyngraph_expression::{Dictionary, Expression, ExpressionAtom, Translator};

use pretty_assertions::assert_eq;

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Eq, Ord)]
struct Symbol<'a>(&'a str);

impl<'a> Display for Symbol<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'a> ExpressionAtom for Symbol<'a> {}

macro_rules! s {
    ($str: ident) => {
        Expression::Atom(Symbol(stringify!($str)))
    };
}
macro_rules! i {
    ($i: expr) => {
        Expression::IConst($i)
    };
}
macro_rules! f {
    ($f: expr) => {
        Expression::FConst($f)
    };
}
macro_rules! inv {
    ($e: expr) => {
        Expression::Inverse(Box::new($e))
    };
}
macro_rules! pow {
    ($x: expr, $e: expr) => {
        Expression::Power(Box::new($x), Box::new($e))
    };
}
macro_rules! func {
    ($name: literal) => {
        Expression::Function {module: None, name: String::from($name), exprs: Vec::new()}
    };
    ($name: literal; $( $e: expr ),+) => {
        Expression::Function {module: None, name: String::from($name), exprs: vec![$($e),+]}
    };
    ($module:literal, $name: literal; $( $e: expr ),+) => {
        Expression::Function {module: Some(String::from($module)), name: String::from($name), exprs: vec![$($e),+]}
    }
}

struct SymbolPyDict;

impl Dictionary<Symbol<'_>, String, SymbolPyTranslator> for SymbolPyDict {
    const POWER: &'static str = "**";

    fn atom(buf: &mut String, atom: &Symbol, _ctx: &mut SymbolPyTranslator) -> Result<(), std::fmt::Error> {
        buf.write_str(atom.0)
    }
}

struct SymbolPyTranslator;

impl Translator<Symbol<'_>, SymbolPyDict> for SymbolPyTranslator {
    type Output = String;

    fn translate(&mut self, expr: &Expression<Symbol<'_>>) -> Result<Self::Output, Error> {
        let mut buf = String::new();
        self.expr(&mut buf, expr)?;
        Ok(buf)
    }
}

#[test]
fn test_symbol_expand() {
    let expr = (s!(x) + s!(y)) * (s!(x) + s!(y));
    assert_eq!(expr.normalize(), s!(x) * s!(x) + i!(2) * s!(x) * s!(y) + s!(y) * s!(y));
    let expr = (s!(x) + s!(y)) * (s!(x) - s!(y));
    assert_eq!(expr.normalize(), (s!(x) * s!(x) - s!(y) * s!(y)).normalize());
    let expr = (s!(x) - s!(y)) * (s!(x) - s!(y));
    assert_eq!(
        expr.normalize(),
        (s!(x) * s!(x) - i!(2) * s!(x) * s!(y) + s!(y) * s!(y)).normalize()
    );
}

#[test]
fn test_symbol_zero_product() {
    let expr: Expression<Symbol> = i!(1) + i!(-1);
    assert_eq!(expr.normalize(), i!(0));
    let expr = i!(0) * s!(x);
    assert_eq!(expr.normalize(), i!(0));
}

#[test]
fn test_symbol_expand_large() {
    let base: Expression<Symbol> = s!(x) + s!(y);
    let res = (0..65)
        .map(|_| base.clone())
        .fold(base.clone(), |x, y| x * y)
        .normalize();
    let n = match res {
        Expression::Sum(x) => match &x[33] {
            Expression::Product(y) => match y[0] {
                Expression::IConst(i) => i,
                _ => panic!(),
            },
            _ => panic!(),
        },
        _ => panic!(),
    };
    assert_eq!(n, 7219428434016265740);
}

#[test]
fn test_symbol_replace() {
    let mut expr = pow!(s!(x), i!(3)) * s!(y) + func!("exp"; s!(i) * f!(3.14) * s!(theta));
    expr.replace(s!(x), s!(xi));
    expr.replace(s!(theta), s!(phi));
    expr.replace(s!(i) * s!(phi), s!(z));
    let expr_ref = pow!(s!(xi), i!(3)) * s!(y) + func!("exp"; f!(3.14) * s!(z));
    assert_eq!(expr, expr_ref);
}

#[test]
fn test_symbol_inv() {
    let expr = i!(3) / i!(2) * inv!(s!(x)) * inv!(s!(y)) * inv!(inv!(s!(z)));
    let expr_ref = f!(3. / 2.) * s!(z) / (s!(x) * s!(y));
    assert_eq!(expr.normalize(), expr_ref);
}

#[test]
fn test_symbol_translate_py() {
    let arg = pow!(f!(4.) * s!(x) - i!(3) * s!(y), s!(z));
    let f = func!("math", "cos"; arg);
    let expr = s!(A) * f + s!(B) * func!("math", "sin"; s!(x));
    assert_eq!(
        SymbolPyTranslator.translate(&expr.normalize()).unwrap(),
        "((A*math.cos((((4*x)+(-3*y)))**(z)))+(B*math.sin(x)))"
    );
}
