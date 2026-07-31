use std::mem::transmute;

use crate::{
    ColorAtom,
    error::ParseError,
    parser::{
        common::{argument_list, atom_arguments},
        diagnostic::{UnknownColorAtom, UnrecoveredParserError},
        expression_ast::ExpressionAst,
        math_base::math_base,
    },
};
use chumsky::prelude::*;
use expression::Expression;
use util::parser::{Diagnostic, ParseResult};

fn atom_parser<'src>()
-> impl Parser<'src, &'src str, ParseResult<ColorAtom>, chumsky::extra::Err<Rich<'src, char>>> + Send + Sync + Clone {
    let atom = choice((
        just("1").map(|_| ParseResult::Ok(ColorAtom::Trivial)),
        atom_arguments("Identity", 2).map(|x| x.map(|v| ColorAtom::Identity(v[0], v[1]))),
        atom_arguments("T", 3).map(|x| x.map(|v| ColorAtom::T(v[0], v[1], v[2]))),
        atom_arguments("f", 3).map(|x| x.map(|v| ColorAtom::f(v[0], v[1], v[2]))),
        atom_arguments("d", 3).map(|x| x.map(|v| ColorAtom::d(v[0], v[1], v[2]))),
        atom_arguments("Epsilon", 3).map(|x| x.map(|v| ColorAtom::Epsilon(v[0], v[1], v[2]))),
        atom_arguments("EpsilonBar", 3).map(|x| x.map(|v| ColorAtom::EpsilonBar(v[0], v[1], v[2]))),
        atom_arguments("T6", 3).map(|x| x.map(|v| ColorAtom::T6(v[0], v[1], v[2]))),
        atom_arguments("K6", 3).map(|x| x.map(|v| ColorAtom::K6(v[0], v[1], v[2]))),
        atom_arguments("K6Bar", 3).map(|x| x.map(|v| ColorAtom::K6Bar(v[0], v[1], v[2]))),
    ))
    .recover_with(via_parser(
        text::ident()
            .spanned()
            .then_ignore(argument_list())
            .map(|s| ParseResult::Err(vec![Box::new(UnknownColorAtom::new(s.inner, s.span))])),
    ));
    return atom;
}

fn parser<'src>() -> impl Parser<'src, &'src str, ExpressionAst<ColorAtom>, chumsky::extra::Err<Rich<'src, char>>> {
    return math_base(atom_parser(), false, false);
}

pub(crate) struct ColorParser<'parser> {
    inner: Box<
        dyn Parser<'parser, &'parser str, ExpressionAst<ColorAtom>, chumsky::extra::Err<Rich<'parser, char>>> + 'parser,
    >,
}

impl<'parser> ColorParser<'parser> {
    pub fn new() -> Self {
        return Self {
            inner: Box::new(parser::<'parser>()),
        };
    }

    pub fn parse<'src>(&self, input: &'src str, name: &str, index: usize) -> Result<Expression<ColorAtom>, ParseError> {
        let input: &'parser str = unsafe { transmute(input) };
        let (result, errs) = self.inner.parse(input).into_output_errors();
        match result {
            None => {
                return Err(ParseError::new(
                    format!("{name}[{index}]"),
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
    use crate::{
        components::{ColorAtom, ColorExpression},
        parser::color::ColorParser,
    };

    #[test]
    fn basic_color_parse_test() {
        let input = "d(-1,1,4)*f(-1,2,3) + d(-1,2,3)*f(-1,1,4)";
        let expr = ColorParser::new().parse(input, "test_color", 0).unwrap();
        let expected = ColorExpression::Sum(vec![
            ColorExpression::Product(vec![
                ColorExpression::Atom(ColorAtom::d(-1, 1, 4)),
                ColorExpression::Atom(ColorAtom::f(-1, 2, 3)),
            ]),
            ColorExpression::Product(vec![
                ColorExpression::Atom(ColorAtom::d(-1, 2, 3)),
                ColorExpression::Atom(ColorAtom::f(-1, 1, 4)),
            ]),
        ]);
        assert_eq!(expr, expected);
    }
}
