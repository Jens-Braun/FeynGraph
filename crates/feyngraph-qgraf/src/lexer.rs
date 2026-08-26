use logos::Logos;
use util::parsing::{PSLexer, Sep};

pub(crate) type Lexer<'s> = PSLexer<'s, Token<'s>>;

fn trim_str<'s>(lex: &mut logos::Lexer<'s, Token<'s>>) -> Option<&'s str> {
    Some(&lex.slice()[1..lex.slice().len() - 1])
}

#[derive(Debug, Clone, Logos)]
#[logos(subpattern id_start = r"[a-zA-Z]")]
#[logos(subpattern id_continue = r"[a-zA-Z0-9_-~+]")]
#[logos(skip r"\s+")]
#[logos(skip(r"%.*\n", allow_greedy = true))]
pub(crate) enum Token<'s> {
    #[regex(r"[+-]?\d[\d_]*", |lex| lex.slice().parse::<i64>().unwrap())]
    Int(i64),
    #[regex(r#"("[^"]*")|('[^']*')"#, trim_str)]
    String(&'s str),
    #[regex(r"(?&id_start)(?&id_continue)*")]
    Ident(&'s str),

    #[token("[")]
    BrackOpen,
    #[token("]")]
    BrackClose,
    #[token("(")]
    ParenOpen,
    #[token(")")]
    ParenClose,

    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,

    #[token("=")]
    Eq,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
}

impl<'s> Token<'s> {
    pub(crate) fn int(self) -> i64 {
        match self {
            Self::Int(i) => return i,
            _ => unreachable!(),
        }
    }
    pub(crate) fn ident(self) -> &'s str {
        match self {
            Self::Ident(s) => return s,
            _ => unreachable!(),
        }
    }
    pub(crate) fn str(self) -> &'s str {
        match self {
            Self::String(s) => return s,
            _ => unreachable!(),
        }
    }
}

impl<'s> Sep for Token<'s> {
    fn is_sep(&self) -> bool {
        match self {
            Self::BrackOpen | Self::BrackClose | Self::ParenOpen | Self::ParenClose | Self::Comma | Self::Semicolon => {
                return true;
            }
            _ => return false,
        }
    }
}
