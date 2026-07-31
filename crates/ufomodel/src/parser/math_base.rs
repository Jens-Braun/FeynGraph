use chumsky::{
    pratt::{infix, left, prefix, right},
    prelude::*,
    text::unicode::ident,
};

use crate::{
    parser::common::{float, uint},
    parser::expression_ast::ExpressionAst,
};
use expression::ExpressionAtom;
use util::parser::ParseResult;

use std::fmt::Debug;

pub(crate) fn math_base<'src, A: ExpressionAtom + Debug + 'src>(
    atom: impl Parser<'src, &'src str, ParseResult<A>, chumsky::extra::Err<Rich<'src, char>>> + Clone + 'src,
    allow_exponents: bool,
    allow_functions: bool,
) -> impl Parser<'src, &'src str, ExpressionAst<A>, chumsky::extra::Err<Rich<'src, char>>> {
    let atom = choice((
        atom.map(|a| match a {
            ParseResult::Ok(a) => ExpressionAst::Atom(a),
            ParseResult::Err(errs) => ExpressionAst::Error(errs),
        }),
        float().try_map(|f, span| match f {
            ParseResult::Err(_) => Err(Rich::custom(span, "invalid float")),
            ParseResult::Ok(f) => Ok(ExpressionAst::FConst(f)),
        }),
        uint().try_map(|i, span| match i {
            ParseResult::Err(_) => Err(Rich::custom(span, "invalid int")),
            ParseResult::Ok(i) => Ok(ExpressionAst::IConst(i)),
        }),
    ))
    .map_err(|e| Rich::custom(*e.span(), "Expected integer, float, operator, function or atom"));

    let op = |char: char| just(char).padded();

    return recursive(|expr| {
        let base_ops = (
            prefix(3, op('-'), |_, rhs: ExpressionAst<A>, _| -rhs),
            infix(left(2), op('*'), |lhs: ExpressionAst<A>, _, rhs, _| lhs * rhs),
            infix(left(2), op('/'), |lhs: ExpressionAst<A>, _, rhs, _| lhs / rhs),
            infix(left(1), op('+'), |lhs: ExpressionAst<A>, _, rhs, _| lhs + rhs),
            infix(left(1), op('-'), |lhs: ExpressionAst<A>, _, rhs, _| lhs - rhs),
        );
        let exponent = infix(right(4), just("**").padded(), |l, _, r, _| {
            ExpressionAst::Power(Box::new(l), Box::new(r))
        });
        let base = atom.or(expr.clone().delimited_by(just("("), just(")")).padded().map(|x| x));
        let base = if allow_functions {
            base.or(ident()
                .then_ignore(just("."))
                .or_not()
                .then(ident())
                .then(
                    expr.separated_by(just(",").padded())
                        .collect()
                        .delimited_by(just("("), just(")")),
                )
                .map(|((module, f), x)| ExpressionAst::Function {
                    module: module.map(|s| s.to_owned()),
                    name: f.to_owned(),
                    exprs: x,
                })
                .padded())
                .boxed()
        } else {
            base.boxed()
        };
        if allow_exponents {
            base.pratt((exponent, base_ops.0, base_ops.1, base_ops.2, base_ops.3, base_ops.4))
                .boxed()
        } else {
            base.pratt(base_ops).boxed()
        }
    });
}
