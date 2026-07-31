use std::mem::transmute;

use crate::{
    LorentzAtom as LAtom,
    error::ParseError,
    parser::{
        common::{argument_list, atom_arguments},
        diagnostic::{UnknownLorentzAtom, UnrecoveredParserError},
        expression_ast::ExpressionAst,
        math_base::math_base,
    },
};
use chumsky::prelude::*;
use expression::Expression;
use util::parser::{Diagnostic, ParseResult};

fn atom_parser<'src>() -> impl Parser<'src, &'src str, ParseResult<LAtom>, chumsky::extra::Err<Rich<'src, char>>> + Clone
{
    let atom = choice((
        just("1").map(|_| ParseResult::Ok(LAtom::Trivial)),
        atom_arguments("Identity", 2).map(|x| x.map(|v| LAtom::Identity(v[0], v[1]))),
        atom_arguments("IdentityL", 2).map(|x| x.map(|v| LAtom::IdentityL(v[0], v[1]))),
        atom_arguments("Gamma", 3).map(|x| x.map(|v| LAtom::Gamma(v[0], v[1], v[2]))),
        atom_arguments("Gamma5", 2).map(|x| x.map(|v| LAtom::Gamma5(v[0], v[1]))),
        atom_arguments("ProjM", 2).map(|x| x.map(|v| LAtom::ProjM(v[0], v[1]))),
        atom_arguments("ProjP", 2).map(|x| x.map(|v| LAtom::ProjP(v[0], v[1]))),
        atom_arguments("Sigma", 4).map(|x| x.map(|v| LAtom::Sigma(v[0], v[1], v[2], v[3]))),
        atom_arguments("C", 2).map(|x| x.map(|v| LAtom::C(v[0], v[1]))),
        atom_arguments("Metric", 2).map(|x| x.map(|v| LAtom::Metric(v[0], v[1]))),
        atom_arguments("P", 2)
            .then_ignore(just("**2"))
            .map(|x| x.map(|v| LAtom::Psq(v[0], v[1]))),
        atom_arguments("P", 2).map(|x| x.map(|v| LAtom::P(v[0], v[1]))),
        atom_arguments("Epsilon", 4).map(|x| x.map(|v| LAtom::Epsilon(v[0], v[1], v[2], v[3]))),
    ))
    .recover_with(via_parser(
        text::ident()
            .spanned()
            .then_ignore(argument_list())
            .map(|s| ParseResult::Err(vec![Box::new(UnknownLorentzAtom::new(s.inner, s.span))])),
    ));
    return atom;
}

fn parser<'src>() -> impl Parser<'src, &'src str, ExpressionAst<LAtom>, chumsky::extra::Err<Rich<'src, char>>> {
    return math_base(atom_parser(), false, false);
}

pub(crate) struct LorentzParser<'parser> {
    inner: Box<
        dyn Parser<'parser, &'parser str, ExpressionAst<LAtom>, chumsky::extra::Err<Rich<'parser, char>>> + 'parser,
    >,
}

impl<'parser> LorentzParser<'parser> {
    pub fn new() -> Self {
        return Self {
            inner: Box::new(parser::<'parser>()),
        };
    }

    pub fn parse<'src>(&self, input: &'src str, name: &str) -> Result<Expression<LAtom>, ParseError> {
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
    use crate::{components::LorentzAtom, parser::lorentz::LorentzParser};
    use expression::Expression;

    #[test]
    fn parse_basic_lorentz_test() {
        let input = "P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)";
        let res = LorentzParser::new().parse(input, "test").unwrap();
        let expected = Expression::Product(vec![
            Expression::Atom(LorentzAtom::P(-1, 1)),
            Expression::Atom(LorentzAtom::Gamma(-1, 2, -2)),
            Expression::Atom(LorentzAtom::ProjM(-2, 1)),
        ]);
        assert_eq!(res, expected);
    }

    #[test]
    fn parse_basic_lorentz_error_test() {
        let input = "P(-1,1)*Gamma(-1,k,-2) + ProM(-2,1)";
        let res = LorentzParser::new().parse(input, "test");
        assert!(res.is_err());
        #[cfg(feature = "emit-test-reports")]
        res.unwrap_err().emit().unwrap();
    }

    #[test]
    fn parse_complex_lorentz_test() {
        let input = "P(-1,1) * -Gamma(-1,2,-2) - -ProjM(-2,1) * - ProjP(-2, 1)";
        let res = LorentzParser::new().parse(input, "test").unwrap();
        let expected = -Expression::Sum(vec![
            Expression::Product(vec![
                Expression::Atom(LorentzAtom::P(-1, 1)),
                Expression::Atom(LorentzAtom::Gamma(-1, 2, -2)),
            ]),
            Expression::Product(vec![
                Expression::Atom(LorentzAtom::ProjM(-2, 1)),
                Expression::Atom(LorentzAtom::ProjP(-2, 1)),
            ]),
        ]);
        assert_eq!(res, expected);
    }

    #[test]
    fn parse_complex_lorentz_2_test() {
        let input =
            "-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*#Gamma(3,2,-3)*ProjM(-2,1)";
        let res = LorentzParser::new().parse(input, "test");
        assert!(res.is_err());
        #[cfg(feature = "emit-test-reports")]
        res.unwrap_err().emit().unwrap();
    }
}
