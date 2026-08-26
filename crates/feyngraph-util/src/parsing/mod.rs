use std::{fmt::Display, ops::Range};

use logos::{Lexer, Logos};

pub mod diagnostic;

pub use diagnostic::{DiagnosticSink, InvalidFloat, InvalidInt, Spanned, UnexpectedEOF};

/// Wrapped Logos lexer which produces `Spanned<Token>` values and allows single-token peeking
pub struct PSLexer<'source, Token: Logos<'source>> {
    inner: Lexer<'source, Token>,
    cache: Option<Option<Spanned<Result<Token, Token::Error>>>>,
}

impl<'source, Token: Logos<'source>> PSLexer<'source, Token> {
    pub fn new(source: &'source Token::Source) -> Self
    where
        Token::Extras: Default,
    {
        Self::with_extras(source, Default::default())
    }

    pub fn with_extras(source: &'source Token::Source, extras: Token::Extras) -> Self {
        Self {
            inner: Lexer::with_extras(source, extras),
            cache: None,
        }
    }

    pub fn peek(&mut self) -> Option<&Spanned<Result<Token, Token::Error>>> {
        if self.cache.is_none() {
            self.cache = Some(self.inner.next().map(|tok| (tok, self.inner.span()).into()));
        }
        self.cache.as_ref().unwrap().as_ref()
    }

    pub fn peek_report(
        &mut self,
        expect: &'static str,
        sink: &mut DiagnosticSink,
    ) -> Option<&Spanned<Result<Token, Token::Error>>> {
        let span = self.span();
        match self.peek() {
            None => {
                sink.push(UnexpectedEOF::expected(expect, span));
                return None;
            }
            tok @ Some(_) => return tok,
        }
    }

    pub fn peek_report_list(
        &mut self,
        expect: &[impl Display],
        sink: &mut DiagnosticSink,
    ) -> Option<&Spanned<Result<Token, Token::Error>>> {
        let span = self.span();
        match self.peek() {
            None => {
                sink.push(UnexpectedEOF::expected_list(expect, span));
                return None;
            }
            tok @ Some(_) => return tok,
        }
    }

    pub fn span(&self) -> Range<usize> {
        self.inner.span()
    }
}

impl<'source, Token: Logos<'source>> Iterator for PSLexer<'source, Token> {
    type Item = Spanned<Result<Token, Token::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.cache.take() {
            Some(tok) => tok,
            None => self.inner.next().map(|tok| (tok, self.inner.span()).into()),
        }
    }
}

pub trait Sep {
    fn is_sep(&self) -> bool;
}

impl<'source, Token: Logos<'source> + Sep> PSLexer<'source, Token> {
    pub fn peek_sep(&mut self) -> bool {
        match self.peek() {
            Some(tok) => match &tok.inner {
                Ok(t) => return t.is_sep(),
                _ => return false,
            },
            _ => return true,
        }
    }
}
