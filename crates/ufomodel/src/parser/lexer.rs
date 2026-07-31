use logos::Logos;

#[derive(Logos, Clone, PartialEq, Debug)]
#[logos(subpattern digipart = r"\d[\d_]*")]
#[logos(subpattern presign = r"\+|-")]
// This is not a real Python lexer, so we just ignore whitespace and only extract object definitions as if this
// was a pure data file format
#[logos(skip r"[ \t]+")]
#[logos(skip r"#.*\n")]
pub enum PythonToken<'src> {
    Error,

    #[regex(r"(?:\r?\n)+")]
    Newline,

    #[regex(r"[[:alpha:]_][[:alpha:]_\d]*")]
    Name(&'src str),
    #[regex(r"(?&presign)?(?&digipart)", |s| s.slice().parse::<i64>().unwrap())]
    Int(i64),
    #[regex(r"(?&presign)?(?&digipart)\.(?&digipart)(?:[eE][\+-]?(?&digipart)])?", |s| s.slice().parse::<f64>().unwrap())]
    Float(f64),
    #[token("True", |_| true)]
    #[token("False", |_| false)]
    Bool(bool),
    #[regex(r#""[^"]*"|'[^']*'"#, |s| &s.slice()[1..(s.slice().len()-1)])]
    String(&'src str),
    #[regex(r#"[rft]"[^"]"|'[^']'"#, priority = 5)]
    ModString(&'src str),

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBrack,
    #[token("]")]
    RBrack,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token(",")]
    Comma,
    #[token(".")]
    Period,
    #[token("=")]
    Equals,
}

#[cfg(test)]
mod tests {
    use logos::Logos;

    use crate::parser::lexer::PythonToken;

    #[test]
    fn basic_lex_test() {
        let str = r"# This file was automatically created by FeynRules 2.3.49
        # Mathematica version: 14.1.0 for Microsoft Windows (64-bit) (July 16, 2024)
        # Date: Fri 22 Nov 2024 22:31:21


        from object_library import all_orders, CouplingOrder


        QCD = CouplingOrder(name = 'QCD',
                            expansion_order = 99,
                            hierarchy = 1)

";
        let tokens = PythonToken::lexer(str).collect::<Result<Vec<_>, _>>();
        assert!(tokens.is_ok());
    }
}
