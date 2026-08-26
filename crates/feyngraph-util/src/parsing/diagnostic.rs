use ariadne::{Color, Fmt, Label, Report, ReportKind, Source};
use std::{
    fmt::{Debug, Display},
    ops::{Deref, Range},
};

pub struct DiagnosticContainer<T> {
    result: Option<T>,
    sink: DiagnosticSink,
    name: String,
    input: String,
}

impl<T> DiagnosticContainer<T> {
    pub fn new(result: Option<T>, sink: DiagnosticSink, name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            result,
            sink,
            name: name.into(),
            input: input.into(),
        }
    }
    pub fn get(self) -> Option<T> {
        self.result
    }
    pub fn is_some(&self) -> bool {
        self.result.is_some()
    }
    pub fn emit(&self) -> std::io::Result<()> {
        self.sink.emit(&self.name, &self.input)
    }
    pub fn emit_err(&self) -> std::io::Result<()> {
        self.sink.emit_err(&self.name, &self.input)
    }
    pub fn emit_warn(&self) -> std::io::Result<()> {
        self.sink.emit_warn(&self.name, &self.input)
    }
}

#[derive(Debug, PartialEq)]
pub enum Kind {
    Error,
    Warning,
    Misc,
}

pub trait Diagnostic: Debug {
    fn build<'a>(&self, name: &'a str) -> Report<'a, (&'a str, Range<usize>)>;

    fn emit(&self, name: &str, src: &str) -> std::io::Result<()> {
        self.build(name).eprint((name, Source::from(src)))
    }

    fn emit_cached(&self, name: &str, src: &Source<&str>) -> std::io::Result<()> {
        self.build(name).eprint((name, src))
    }

    fn kind(&self) -> Kind {
        Kind::Error
    }

    fn try_merge_expected(&mut self, span: &Range<usize>, expected: &[String]) -> bool;
    fn expected(&self) -> Option<(&Range<usize>, &[String])>;
}

pub struct DiagnosticSink {
    diagnostics: Vec<Box<dyn Diagnostic>>,
    checkpoint: Option<usize>,
}

impl DiagnosticSink {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            checkpoint: None,
        }
    }

    pub fn push(&mut self, err: impl Diagnostic + 'static) {
        if let Some(e) = self.diagnostics.last_mut()
            && let Some((span, new)) = err.expected()
        {
            if e.try_merge_expected(span, new) {
                return;
            }
        }
        self.diagnostics.push(Box::new(err) as Box<dyn Diagnostic>);
    }

    pub fn set_checkpoint(&mut self) {
        self.checkpoint = Some(self.diagnostics.len());
    }

    pub fn restore_checkpoint(&mut self) {
        if let Some(n) = self.checkpoint {
            self.diagnostics.truncate(n);
        }
        self.checkpoint = None;
    }

    pub fn emit(&self, name: &str, input: &str) -> std::io::Result<()> {
        let src = Source::from(input);
        for err in self.diagnostics.iter() {
            err.emit_cached(name, &src)?;
        }
        Ok(())
    }

    pub fn emit_err(&self, name: &str, input: &str) -> std::io::Result<()> {
        let src = Source::from(input);
        for err in self.diagnostics.iter() {
            if err.kind() == Kind::Error {
                err.emit_cached(name, &src)?;
            }
        }
        Ok(())
    }

    pub fn emit_warn(&self, name: &str, input: &str) -> std::io::Result<()> {
        let src = Source::from(input);
        for err in self.diagnostics.iter() {
            if err.kind() == Kind::Warning {
                err.emit_cached(name, &src)?;
            }
        }
        Ok(())
    }

    pub fn emit_misc(&self, name: &str, input: &str) -> std::io::Result<()> {
        let src = Source::from(input);
        for err in self.diagnostics.iter() {
            if err.kind() == Kind::Misc {
                err.emit_cached(name, &src)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Spanned<T> {
    pub inner: T,
    pub span: Range<usize>,
}

impl<T> Spanned<T> {
    pub fn tup(&self) -> (Range<usize>, &T) {
        (self.span.clone(), &self.inner)
    }
}

impl<T> From<(T, Range<usize>)> for Spanned<T> {
    fn from(value: (T, Range<usize>)) -> Self {
        Self {
            inner: value.0,
            span: value.1,
        }
    }
}

impl<T, U: PartialEq<T>> PartialEq<U> for Spanned<T> {
    fn eq(&self, other: &U) -> bool {
        other.eq(&self.inner)
    }
}

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// -------------------- Errors --------------------

#[derive(Debug)]
pub struct UnexpectedEOF {
    expected: Option<Vec<String>>,
    span: Range<usize>,
}

impl UnexpectedEOF {
    pub fn new(span: Range<usize>) -> Self {
        Self { expected: None, span }
    }

    pub fn expected(expected: impl Into<String>, span: Range<usize>) -> Self {
        Self {
            expected: Some(vec![expected.into().fg(Color::Red).to_string()]),
            span,
        }
    }

    pub fn expected_list(expected: &[impl Display], span: Range<usize>) -> Self {
        Self {
            expected: Some(
                expected
                    .iter()
                    .map(|s| s.fg(Color::Red).to_string())
                    .collect::<Vec<_>>(),
            ),
            span,
        }
    }
}

impl Diagnostic for UnexpectedEOF {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        let expected = match &self.expected {
            None => None,
            Some(v) if v.is_empty() => None,
            Some(v) if v.len() == 1 => Some(v[0].clone()),
            Some(v) => {
                let mut buf = v[..v.len() - 1].join(", ");
                buf.push_str(" or ");
                buf.push_str(v.last().unwrap());
                Some(buf)
            }
        };
        let mut rep = Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(1)
            .with_message("Unexpected end of input");
        if let Some(exp) = expected {
            rep.add_label(
                Label::new((name, self.span.clone()))
                    .with_message(format!("Expected {}", exp))
                    .with_color(Color::Red),
            );
        }
        rep.finish()
    }

    fn try_merge_expected(&mut self, span: &Range<usize>, expected: &[String]) -> bool {
        if *span == self.span
            && let Some(exp) = &mut self.expected
        {
            exp.extend_from_slice(expected);
            exp.dedup();
        } else {
            self.expected = Some(expected.to_vec());
        }
        true
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        self.expected.as_ref().map(|v| (&self.span, v.as_slice()))
    }
}

#[derive(Debug)]
pub struct UnexpectedInput {
    expected: Option<Vec<String>>,
    span: Range<usize>,
}

impl UnexpectedInput {
    pub fn new(span: Range<usize>) -> Self {
        Self { expected: None, span }
    }

    pub fn expected(expected: impl Into<String>, span: Range<usize>) -> Self {
        Self {
            expected: Some(vec![expected.into().fg(Color::Red).to_string()]),
            span,
        }
    }

    pub fn expected_list(expected: &[impl Display], span: Range<usize>) -> Self {
        Self {
            expected: Some(
                expected
                    .iter()
                    .map(|s| s.fg(Color::Red).to_string())
                    .collect::<Vec<_>>(),
            ),
            span,
        }
    }
}

impl Diagnostic for UnexpectedInput {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        let expected = match &self.expected {
            None => None,
            Some(v) if v.is_empty() => None,
            Some(v) if v.len() == 1 => Some(v[0].clone()),
            Some(v) => {
                let mut buf = v[..v.len() - 1].join(", ");
                buf.push_str(" or ");
                buf.push_str(v.last().unwrap());
                Some(buf)
            }
        };
        let mut rep = Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(2)
            .with_message("Unexpected input");
        if let Some(exp) = expected {
            rep.add_label(
                Label::new((name, self.span.clone()))
                    .with_message(format!("Expected {}", exp))
                    .with_color(Color::Red),
            );
        }
        rep.finish()
    }

    fn try_merge_expected(&mut self, span: &Range<usize>, expected: &[String]) -> bool {
        if *span == self.span
            && let Some(exp) = &mut self.expected
        {
            exp.extend_from_slice(expected);
            exp.dedup();
        } else {
            self.expected = Some(expected.to_vec());
        }
        true
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        self.expected.as_ref().map(|v| (&self.span, v.as_slice()))
    }
}

#[derive(Debug)]
pub struct InvalidInt {
    span: Range<usize>,
}

impl InvalidInt {
    pub fn new(span: Range<usize>) -> Self {
        return Self { span };
    }
}

impl Diagnostic for InvalidInt {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(3)
            .with_message("Expected integer")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message("Unable to interpret this expression as integer")
                    .with_color(Color::Red),
            )
            .with_help(format!("The expected format is {}", "[+-]?[0-9]+".fg(Color::Cyan)))
            .finish()
    }

    fn try_merge_expected(&mut self, _span: &Range<usize>, _expected: &[String]) -> bool {
        false
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        None
    }
}

#[derive(Debug)]
pub struct InvalidFloat {
    span: Range<usize>,
}

impl InvalidFloat {
    pub fn new(span: Range<usize>) -> Self {
        return Self { span };
    }
}

impl Diagnostic for InvalidFloat {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(4)
            .with_message("Expected float")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message("Unable to interpret this expression as float")
                    .with_color(Color::Red),
            )
            .with_help(format!(
                "The expected format is {}",
                "[+-]?[0-9]*.[0-9]*([eE][0-9]+)".fg(Color::Cyan)
            ))
            .finish()
    }

    fn try_merge_expected(&mut self, _span: &Range<usize>, _expected: &[String]) -> bool {
        false
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        None
    }
}

#[derive(Debug)]
pub struct InvalidString {
    span: Range<usize>,
}

impl InvalidString {
    pub fn new(span: Range<usize>) -> Self {
        return Self { span };
    }
}

impl Diagnostic for InvalidString {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(4)
            .with_message("Expected string")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message("Unable to interpret this expression as valid string")
                    .with_color(Color::Red),
            )
            .with_help(format!(
                "The expected format is {}",
                r#"("[^"]*")|('[^']*')"#.fg(Color::Cyan)
            ))
            .finish()
    }

    fn try_merge_expected(&mut self, _span: &Range<usize>, _expected: &[String]) -> bool {
        false
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        None
    }
}
