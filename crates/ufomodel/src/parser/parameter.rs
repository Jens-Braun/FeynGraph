use std::mem::transmute;

use crate::{
    ParameterAtom,
    error::ParseError,
    parser::{diagnostic::UnrecoveredParserError, expression_ast::ExpressionAst, math_base::math_base},
};
use chumsky::{prelude::*, text::ident};
use expression::Expression;
use num_complex::Complex;
use util::parser::{Diagnostic, ParseResult};

pub enum RawParameterValue {
    ExternalReal(f64),
    ExternalComplex(Complex<f64>),
    InternalReal(String),
    InternalComplex(String),
}

fn atom_parser<'src>()
-> impl Parser<'src, &'src str, ParseResult<ParameterAtom>, chumsky::extra::Err<Rich<'src, char>>> + Send + Sync + Clone
{
    let atom = choice((
        just("math")
            .or(just("cmath"))
            .then(just("."))
            .ignore_then(ident())
            .then_ignore(none_of("(").rewind().ignored().or(end()))
            .map(|c: &str| ParseResult::Ok(ParameterAtom::Constant(c.to_owned()))),
        ident()
            .then_ignore(none_of(".(").rewind().ignored().or(end()))
            .map(|c: &str| ParseResult::Ok(ParameterAtom::Parameter(c.to_owned()))),
        just("complex")
            .then(
                just("0")
                    .padded()
                    .then(just(","))
                    .padded()
                    .then(just("1"))
                    .padded()
                    .delimited_by(just("("), just(")")),
            )
            .map(|_| ParseResult::Ok(ParameterAtom::I)),
    ));
    return atom;
}

fn parser<'src>() -> impl Parser<'src, &'src str, ExpressionAst<ParameterAtom>, chumsky::extra::Err<Rich<'src, char>>> {
    return math_base(atom_parser(), true, true);
}

pub(crate) struct ParameterParser<'parser> {
    inner: Box<
        dyn Parser<'parser, &'parser str, ExpressionAst<ParameterAtom>, chumsky::extra::Err<Rich<'parser, char>>>
            + 'parser,
    >,
}

impl<'parser> ParameterParser<'parser> {
    pub fn new() -> Self {
        return Self {
            inner: Box::new(parser::<'parser>()),
        };
    }

    pub fn parse<'src>(&self, input: &'src str, name: &str) -> Result<Expression<ParameterAtom>, ParseError> {
        let input: &'parser str = unsafe { transmute(input) };
        let (result, errs) = self.inner.parse(input).into_output_errors();
        match result {
            None => {
                return Err(ParseError::new(
                    name,
                    input,
                    errs.into_iter()
                        .map(|e| Box::new(UnrecoveredParserError::new(e.to_string(), *e.span())) as Box<dyn Diagnostic>)
                        .collect(),
                ));
            }
            Some(ast) => return ast.into_result(name, input),
        }
    }
}

#[cfg(test)]
mod tests {
    use expression::Expression;

    use crate::{components::ParameterAtom, parser::parameter::ParameterParser};

    #[test]
    fn basic_parameter_parse_test() {
        let input = "cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))";
        let parser = ParameterParser::new();
        let res = parser.parse(input, "P1").unwrap();
        let reference = Expression::Function {
            module: Some("cmath".into()),
            name: "sqrt".into(),
            exprs: vec![Expression::Sum(vec![
                Expression::Product(vec![
                    Expression::FConst(0.5),
                    Expression::Power(
                        Box::new(Expression::Atom(ParameterAtom::Parameter("MZ".into()))),
                        Box::new(Expression::IConst(2)),
                    ),
                ]),
                Expression::Function {
                    module: Some("cmath".into()),
                    name: "sqrt".into(),
                    exprs: vec![Expression::Sum(vec![
                        Expression::Product(vec![
                            Expression::FConst(0.25),
                            Expression::Power(
                                Box::new(Expression::Atom(ParameterAtom::Parameter("MZ".into()))),
                                Box::new(Expression::IConst(4)),
                            ),
                        ]),
                        Expression::Product(vec![
                            Expression::IConst(-1),
                            Expression::Product(vec![
                                Expression::Atom(ParameterAtom::Parameter("aEW".into())),
                                Expression::Atom(ParameterAtom::Constant("pi".into())),
                                Expression::Power(
                                    Box::new(Expression::Atom(ParameterAtom::Parameter("MZ".into()))),
                                    Box::new(Expression::IConst(2)),
                                ),
                            ]),
                            Expression::Inverse(Box::new(Expression::Product(vec![
                                Expression::Atom(ParameterAtom::Parameter("Gf".into())),
                                Expression::Function {
                                    module: Some("cmath".into()),
                                    name: "sqrt".into(),
                                    exprs: vec![Expression::IConst(2)],
                                },
                            ]))),
                        ]),
                    ])],
                },
            ])],
        };
        assert_eq!(res.normalize(), reference.normalize());
    }
}
