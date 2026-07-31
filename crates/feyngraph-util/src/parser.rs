use ariadne::{Report, Source};
use std::{fmt::Debug, ops::Range};

pub trait Diagnostic: Debug {
    fn build<'a>(&self, name: &'a str) -> Report<'a, (&'a str, Range<usize>)>;

    fn emit(&self, name: &str, src: &str) -> std::io::Result<()> {
        self.build(name).eprint((name, Source::from(src)))
    }

    fn emit_cached(&self, name: &str, src: &Source<&str>) -> std::io::Result<()> {
        self.build(name).eprint((name, src))
    }
}

#[derive(Debug)]
pub enum ParseResult<T> {
    Ok(T),
    Err(Vec<Box<dyn Diagnostic>>),
}

impl<T> ParseResult<T> {
    pub fn is_err(&self) -> bool {
        match self {
            Self::Err { .. } => true,
            Self::Ok(_) => false,
        }
    }

    pub fn unwrap(self) -> T {
        match self {
            Self::Ok(x) => x,
            Self::Err(_) => panic!(),
        }
    }

    #[allow(unused)]
    pub fn emit(&self, name: &str, input: &str) -> std::io::Result<()> {
        match self {
            Self::Ok(_) => (),
            Self::Err(errs) => {
                let src = Source::from(input);
                for err in errs {
                    err.emit_cached(name, &src)?;
                }
            }
        }
        Ok(())
    }
}

impl<T> ParseResult<T> {
    pub fn map<U, F>(self, f: F) -> ParseResult<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Self::Err(errs) => ParseResult::Err(errs),
            Self::Ok(x) => ParseResult::Ok(f(x)),
        }
    }
}
