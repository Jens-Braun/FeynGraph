use chumsky::prelude::*;

use crate::parser::diagnostic::{InvalidAtomArgumentNumber, InvalidFloatDigit, InvalidIntDigit};
use util::parser::ParseResult;

pub(crate) fn uint<'src>()
-> impl Parser<'src, &'src str, ParseResult<i64>, chumsky::extra::Err<Rich<'src, char>>> + Clone {
    any()
        .filter(move |c: &char| c.is_digit(10))
        .map(|c| ParseResult::Ok(c))
        .recover_with(via_parser(none_of(",)+-*/ ").map_with(|c: char, e| {
            ParseResult::Err(vec![Box::new(InvalidIntDigit::new(c, e.span()))])
        })))
        .padded()
        .repeated()
        .at_least(1)
        .collect()
        .map(|s: Vec<ParseResult<char>>| {
            if !s.iter().any(|r| r.is_err()) {
                let s = s.into_iter().map(|c| c.unwrap()).collect::<String>();
                return ParseResult::Ok(s.parse::<i64>().unwrap());
            } else {
                let errs: Vec<Box<_>> = s
                    .into_iter()
                    .filter_map(|r| match r {
                        ParseResult::Ok(_) => None,
                        ParseResult::Err(e) => Some(e),
                    })
                    .flatten()
                    .collect();
                return ParseResult::Err(errs.into_iter().collect());
            }
        })
        .labelled("unsigned int")
}

pub(crate) fn sint<'src>()
-> impl Parser<'src, &'src str, ParseResult<i64>, chumsky::extra::Err<Rich<'src, char>>> + Clone {
    return just("-").padded().repeated().foldr(uint(), |_, rhs| rhs.map(|x| -x));
}

// Based on the Python float definition (https://docs.python.org/3/reference/lexical_analysis.html#floating-point-literals)
pub(crate) fn float<'src>()
-> impl Parser<'src, &'src str, ParseResult<f64>, chumsky::extra::Err<Rich<'src, char>>> + Clone {
    let digit = one_of("0123456789").to_slice();
    let digipart = digit.then(just("_").or(digit).repeated());
    let exponent = one_of("eE").then(one_of("+-").or_not()).then(digipart);
    return choice((
        digipart
            .then(just(".").then(digipart.or_not()).then(exponent.or_not()))
            .to_slice(),
        just(".").then(digipart).then(exponent.or_not()).to_slice(),
        digipart.then(exponent).to_slice(),
    ))
    .map(|s: &str| ParseResult::Ok(s.replace("_", "").parse::<f64>().unwrap()))
    .recover_with(via_parser(none_of(",)+-*/ ").map_with(|c: char, e| {
        ParseResult::Err(vec![Box::new(InvalidFloatDigit::new(c, e.span()))])
    })))
    .labelled("float");
}

pub(crate) fn atom_arguments<'src>(
    s: &'static str,
    n: usize,
) -> impl Parser<'src, &'src str, ParseResult<Vec<i64>>, chumsky::extra::Err<Rich<'src, char>>> + Clone {
    return text::ident()
        .filter(move |ident| if s.len() > 0 { *ident == s } else { true })
        .ignore_then(argument_list().spanned())
        .map(move |args| {
            if let ParseResult::Ok(v) = args.inner {
                if v.len() == n {
                    return ParseResult::Ok(v);
                } else {
                    return ParseResult::Err(vec![Box::new(InvalidAtomArgumentNumber::new(s, n, v.len(), args.span))]);
                }
            } else {
                return args.inner;
            }
        });
}

pub(crate) fn argument_list<'src>()
-> impl Parser<'src, &'src str, ParseResult<Vec<i64>>, chumsky::extra::Err<Rich<'src, char>>> + Clone {
    sint()
        .separated_by(just(","))
        .collect::<Vec<_>>()
        .map(|s| {
            if !s.iter().any(|r| r.is_err()) {
                return ParseResult::Ok(s.into_iter().map(|i| i.unwrap()).collect());
            } else {
                let errs: Vec<Box<_>> = s
                    .into_iter()
                    .filter_map(|r| match r {
                        ParseResult::Ok(_) => None,
                        ParseResult::Err(e) => Some(e),
                    })
                    .flatten()
                    .collect();
                return ParseResult::Err(errs.into_iter().collect());
            }
        })
        .delimited_by(just("("), just(")"))
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use crate::parser::common::float;

    use super::{atom_arguments, sint, uint};
    use chumsky::prelude::*;

    #[test]
    fn parse_int_test() {
        let parser = sint();
        assert_eq!(-1249931, parser.parse("-----1249931").unwrap().unwrap());
        assert_eq!(-1249931, parser.parse("-1249931").unwrap().unwrap());
        assert_eq!(631934, parser.parse("----631934").unwrap().unwrap());
    }

    #[test]
    fn parse_int_error_test() {
        #[allow(unused)]
        let res = uint().parse("18xu").into_output().unwrap();
        #[cfg(feature = "emit-test-reports")]
        res.emit("test", "18xu").unwrap();
    }

    #[test]
    fn parse_atom_argument_test() {
        let parser = atom_arguments("test_atom", 3);
        assert_eq!(vec![-5, 22, 7], parser.parse("test_atom(-5, 22, 7)").unwrap().unwrap());
        let parser = atom_arguments("P", 2);
        assert_eq!(vec![-1, 1], parser.parse("P(-1,1)").unwrap().unwrap());
    }

    #[test]
    fn parse_atom_error_test() {
        let input = "test_atom(-5, 3, 4, 5)";
        let parser = atom_arguments("test_atom", 3);
        #[allow(unused)]
        let res = parser.parse(input).into_output().unwrap();
        #[cfg(feature = "emit-test-reports")]
        res.emit("test", input).unwrap();
    }

    #[test]
    fn parse_float_test() {
        let input = "14.13e-2";
        let res = float().parse(input).unwrap().unwrap();
        assert_eq!(res, 14.13e-2);
    }
}
