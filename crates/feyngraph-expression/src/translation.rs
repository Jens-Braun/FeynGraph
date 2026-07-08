use crate::{Expression, ExpressionAtom};
use std::fmt::{Error, Write};

pub trait Translator<A: ExpressionAtom, D: Dictionary<A, Self::Output, Self>> {
    type Output: Write;

    fn translate(&mut self, expr: &Expression<A>) -> Result<Self::Output, Error>;

    fn expr(&mut self, buf: &mut Self::Output, expr: &Expression<A>) -> Result<(), Error> {
        match expr {
            Expression::IConst(i) => self.iconst(buf, *i),
            Expression::FConst(f) => self.fconst(buf, *f),
            Expression::Atom(atom) => self.atom(buf, atom),
            Expression::Power(x, e) => self.power(buf, x, e),
            Expression::Inverse(expr) => self.inverse(buf, expr),
            Expression::Sum(exprs) => self.sum(buf, exprs),
            Expression::Product(exprs) => self.product(buf, exprs),
            Expression::Function { module, name, exprs } => self.function(buf, module, name, exprs),
        }
    }

    fn iconst(&mut self, buf: &mut Self::Output, i: i64) -> Result<(), Error> {
        write!(buf, "{}", i)
    }

    fn fconst(&mut self, buf: &mut Self::Output, f: f64) -> Result<(), Error> {
        write!(buf, "{}", f)
    }

    fn atom(&mut self, buf: &mut Self::Output, atom: &A) -> Result<(), Error> {
        D::atom(buf, atom, self)
    }

    fn power(&mut self, buf: &mut Self::Output, x: &Expression<A>, e: &Expression<A>) -> Result<(), Error> {
        buf.write_str(D::L_PAREN)?;
        self.expr(buf, x)?;
        buf.write_str(D::R_PAREN)?;
        buf.write_str(D::POWER)?;
        buf.write_str(D::L_PAREN)?;
        self.expr(buf, e)?;
        buf.write_str(D::R_PAREN)
    }

    fn inverse(&mut self, buf: &mut Self::Output, expr: &Expression<A>) -> Result<(), Error> {
        buf.write_str(D::DIV)?;
        buf.write_str(D::L_PAREN)?;
        self.expr(buf, expr)?;
        buf.write_str(D::R_PAREN)
    }

    fn sum(&mut self, buf: &mut Self::Output, exprs: &[Expression<A>]) -> Result<(), Error> {
        buf.write_str(D::L_PAREN)?;
        for (i, expr) in exprs.iter().enumerate() {
            if i != 0 {
                buf.write_str(D::SUM)?;
            }
            self.expr(buf, expr)?;
        }
        buf.write_str(D::R_PAREN)
    }

    fn product(&mut self, buf: &mut Self::Output, exprs: &[Expression<A>]) -> Result<(), Error> {
        buf.write_str(D::L_PAREN)?;
        for (i, expr) in exprs.iter().enumerate() {
            if i != 0 {
                buf.write_str(D::MULT)?;
            }
            self.expr(buf, expr)?;
        }
        buf.write_str(D::R_PAREN)
    }

    fn function(
        &mut self,
        buf: &mut Self::Output,
        module: &Option<String>,
        name: &str,
        exprs: &[Expression<A>],
    ) -> Result<(), Error> {
        if let Some(module) = module {
            buf.write_str(module)?;
            buf.write_str(D::MODULE_ACCESS)?;
        }
        buf.write_str(name)?;
        buf.write_str(D::L_PAREN)?;
        for (i, expr) in exprs.iter().enumerate() {
            self.expr(buf, expr)?;
            if i != 0 {
                buf.write_str(D::SEP)?;
            }
        }
        buf.write_str(D::R_PAREN)
    }
}

pub trait Dictionary<A: ExpressionAtom, O, C: ?Sized> {
    const L_PAREN: &'static str = "(";
    const R_PAREN: &'static str = ")";
    const SUM: &'static str = "+";
    const SUB: &'static str = "-";
    const MULT: &'static str = "*";
    const DIV: &'static str = "/";
    const POWER: &'static str = "^";
    const MODULE_ACCESS: &'static str = ".";
    const SEP: &'static str = ",";

    fn atom(buf: &mut O, atom: &A, ctx: &mut C) -> Result<(), Error>;
}
