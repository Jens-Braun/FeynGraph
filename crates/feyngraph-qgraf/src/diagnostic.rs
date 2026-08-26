use std::ops::Range;

use ariadne::{Color, Fmt, Label, Report, ReportKind};
use util::parsing::diagnostic::Diagnostic;

// -------------------- Errors --------------------

#[derive(Debug)]
pub struct InvalidIdent {
    span: Range<usize>,
}

impl InvalidIdent {
    pub fn new(span: Range<usize>) -> Self {
        return Self { span };
    }
}

impl Diagnostic for InvalidIdent {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(3)
            .with_message("Expected identifier")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message("This expression is not a valid identifier")
                    .with_color(Color::Red),
            )
            .with_help(format!(
                "The allowed characters for an identifier are {}",
                "[a-zA-Z0-9_-~+]".fg(Color::Cyan)
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
pub(crate) struct UnclosedList {
    start: Range<usize>,
    end: Range<usize>,
}

impl UnclosedList {
    pub fn new(start: Range<usize>, end: Range<usize>) -> Self {
        return Self { start, end };
    }
}

impl Diagnostic for UnclosedList {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.end.clone()))
            .with_code(11)
            .with_message("Unclosed list")
            .with_label(
                Label::new((name, self.start.clone()))
                    .with_message("List begins here")
                    .with_color(Color::Cyan),
            )
            .with_label(
                Label::new((name, self.end.clone()))
                    .with_message(format!("Expected list to end before statement ends here"))
                    .with_color(Color::Red),
            )
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
pub(crate) struct UnclosedStatement {
    start: Range<usize>,
    end: Range<usize>,
}

impl UnclosedStatement {
    pub fn new(start: Range<usize>, end: Range<usize>) -> Self {
        return Self { start, end };
    }
}

impl Diagnostic for UnclosedStatement {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.end.clone()))
            .with_code(12)
            .with_message("Unfinished statement")
            .with_label(
                Label::new((name, self.start.clone()))
                    .with_message("Statement begins here")
                    .with_color(Color::Cyan),
            )
            .with_label(
                Label::new((name, self.end.clone()))
                    .with_message(format!(
                        "Expected statement to end before next statement begins or input ends here"
                    ))
                    .with_color(Color::Red),
            )
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
pub struct InvalidThird {
    span: Range<usize>,
}

impl InvalidThird {
    pub fn new(span: Range<usize>) -> Self {
        return Self { span };
    }
}

impl Diagnostic for InvalidThird {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(13)
            .with_message("Invalid third statement argument")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message("This expression is not an field sign or identifier")
                    .with_color(Color::Red),
            )
            .with_help(
                "The third argument of a statement must be ±(1) to signal a field definition or an identifier to signal a vertex"
            )
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
pub struct UndefinedParticle {
    span: Range<usize>,
    particle: String,
}

impl UndefinedParticle {
    pub fn new(particle: impl Into<String>, span: Range<usize>) -> Self {
        return Self {
            particle: particle.into(),
            span,
        };
    }
}

impl Diagnostic for UndefinedParticle {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.clone()))
            .with_code(14)
            .with_message("Undefined particle")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message(format!("Particle {} is unknown", (&self.particle).fg(Color::Red)))
                    .with_color(Color::Red),
            )
            .with_help("All particles of a vertex must be defined before the vertex")
            .finish()
    }

    fn try_merge_expected(&mut self, _span: &Range<usize>, _expected: &[String]) -> bool {
        false
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        None
    }
}

// -------------------- Warnings --------------------

#[derive(Debug)]
pub struct UnknownStatement {
    span: Range<usize>,
}

impl UnknownStatement {
    pub fn new(span: Range<usize>) -> Self {
        return Self { span };
    }
}

impl Diagnostic for UnknownStatement {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Warning, (name, self.span.clone()))
            .with_code(10)
            .with_message("Unknown statement: ignoring")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message("This expression is not a valid field or vertex definition")
                    .with_color(Color::Yellow),
            )
            .with_help("Only field and vertex definitions are treated as valid statements")
            .finish()
    }

    fn kind(&self) -> util::parsing::diagnostic::Kind {
        util::parsing::diagnostic::Kind::Warning
    }
    fn try_merge_expected(&mut self, _span: &Range<usize>, _expected: &[String]) -> bool {
        false
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        None
    }
}

#[derive(Debug)]
pub struct IntConversion {
    span: Range<usize>,
}

impl IntConversion {
    pub fn new(span: Range<usize>) -> Self {
        return Self { span };
    }
}

impl Diagnostic for IntConversion {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Warning, (name, self.span.clone()))
            .with_code(11)
            .with_message("Automatic string-to-int conversion")
            .with_label(
                Label::new((name, self.span.clone()))
                    .with_message("This string contains only a valid integer and was converted automatically")
                    .with_color(Color::Yellow),
            )
            .with_note("This warning will be shown only once and suppressed for further conversions.")
            .finish()
    }

    fn kind(&self) -> util::parsing::diagnostic::Kind {
        util::parsing::diagnostic::Kind::Warning
    }
    fn try_merge_expected(&mut self, _span: &Range<usize>, _expected: &[String]) -> bool {
        false
    }
    fn expected(&self) -> Option<(&Range<usize>, &[String])> {
        None
    }
}
