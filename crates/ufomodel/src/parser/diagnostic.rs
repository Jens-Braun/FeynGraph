use ariadne::{Color, Fmt, Label, Report, ReportKind};
use chumsky::span::SimpleSpan;
use std::{fmt::Debug, ops::Range};

use util::{HashMap, parser::Diagnostic, unicode::index_sub};

// ---------------------------------- Errors ----------------------------------

#[derive(Debug)]
pub struct UnrecoveredParserError {
    message: String,
    span: SimpleSpan,
}

impl UnrecoveredParserError {
    pub(crate) fn new(message: impl Into<String>, span: SimpleSpan) -> Self {
        return Self {
            message: message.into(),
            span,
        };
    }
}

impl Diagnostic for UnrecoveredParserError {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.into_range()))
            .with_code(0)
            .with_message(&self.message)
            .with_label(
                Label::new((name, self.span.into_range()))
                    .with_message(&self.message)
                    .with_color(Color::Red),
            )
            .finish()
    }
}

#[derive(Debug)]
pub struct InvalidIntDigit {
    found: char,
    span: SimpleSpan,
}

impl InvalidIntDigit {
    pub(crate) fn new(found: char, span: SimpleSpan) -> Self {
        return Self { found, span };
    }
}

impl Diagnostic for InvalidIntDigit {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.into_range()))
            .with_code(1)
            .with_message("Unexpected integer character")
            .with_label(
                Label::new((name, self.span.into_range()))
                    .with_message(format!("Unexpected integer character '{}'", self.found.fg(Color::Red)))
                    .with_color(Color::Red),
            )
            .with_help(format!("The expected format is {}", "[+-]?[0-9]+".fg(Color::Cyan)))
            .finish()
    }
}

#[derive(Debug)]
pub struct InvalidFloatDigit {
    found: char,
    span: SimpleSpan,
}

impl InvalidFloatDigit {
    pub(crate) fn new(found: char, span: SimpleSpan) -> Self {
        return Self { found, span };
    }
}

impl Diagnostic for InvalidFloatDigit {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.into_range()))
            .with_code(2)
            .with_message("Unexpected float character")
            .with_label(
                Label::new((name, self.span.into_range()))
                    .with_message(format!("Unexpected float character '{}'", self.found.fg(Color::Red)))
                    .with_color(Color::Red),
            )
            .with_help(format!(
                "The expected format is {}",
                "[+-]?[0-9]*.[0-9]*([eE][0-9]+)".fg(Color::Cyan)
            ))
            .finish()
    }
}

#[derive(Debug)]
pub struct InvalidAtomArgumentNumber {
    atom: String,
    found: usize,
    expected: usize,
    span: SimpleSpan,
}

impl InvalidAtomArgumentNumber {
    pub(crate) fn new(atom: impl Into<String>, expected: usize, found: usize, span: SimpleSpan) -> Self {
        return Self {
            atom: atom.into(),
            expected,
            found,
            span,
        };
    }
}

impl Diagnostic for InvalidAtomArgumentNumber {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.into_range()))
            .with_code(3)
            .with_message("Unexpected number of atom arguments")
            .with_label(
                Label::new((name, self.span.into_range()))
                    .with_message(format!(
                        "Unexpected number of arguments: '{}'",
                        self.found.fg(Color::Red)
                    ))
                    .with_color(Color::Red),
            )
            .with_help(format!(
                "Atom '{}' expects exactly {} integer arguments",
                (&self.atom).fg(Color::Cyan),
                self.expected.fg(Color::Cyan)
            ))
            .finish()
    }
}

#[derive(Debug)]
pub struct UnknownLorentzAtom {
    found: String,
    span: SimpleSpan,
}

impl UnknownLorentzAtom {
    pub(crate) fn new(found: impl Into<String>, span: SimpleSpan) -> Self {
        return Self {
            found: found.into(),
            span,
        };
    }
}

impl Diagnostic for UnknownLorentzAtom {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.into_range()))
            .with_code(4)
            .with_message("Unexpected Lorentz atom")
            .with_label(Label::new((name, self.span.into_range())).with_message(format!(
                "Unknown Lorentz atom '{}'",
                self.found.clone().fg(Color::Red)
            )).with_color(Color::Red))
            .with_help(format!(
                "Known Lorentz atoms are '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}' and '{}'",
                "1".fg(Color::Cyan),
                "Identity".fg(Color::Cyan),
                "IdentityL".fg(Color::Cyan),
                "Gamma".fg(Color::Cyan),
                "Gamma5".fg(Color::Cyan),
                "ProjM".fg(Color::Cyan),
                "ProjP".fg(Color::Cyan),
                "Sigma".fg(Color::Cyan),
                "C".fg(Color::Cyan),
                "Metric".fg(Color::Cyan),
                "P".fg(Color::Cyan),
                "Psq".fg(Color::Cyan),
                "Epsilon".fg(Color::Cyan),
            ))
            .finish()
    }
}

#[derive(Debug)]
pub struct UnknownColorAtom {
    found: String,
    span: SimpleSpan,
}

impl UnknownColorAtom {
    pub(crate) fn new(found: impl Into<String>, span: SimpleSpan) -> Self {
        return Self {
            found: found.into(),
            span,
        };
    }
}

impl Diagnostic for UnknownColorAtom {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.into_range()))
            .with_code(4)
            .with_message("Unexpected color atom")
            .with_label(
                Label::new((name, self.span.into_range()))
                    .with_message(format!("Unknown color atom '{}'", self.found.clone().fg(Color::Red)))
                    .with_color(Color::Red),
            )
            .with_help(format!(
                "Known color atoms are '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}' and '{}'",
                "1".fg(Color::Cyan),
                "Identity".fg(Color::Cyan),
                "T".fg(Color::Cyan),
                "f".fg(Color::Cyan),
                "d".fg(Color::Cyan),
                "Epsilon".fg(Color::Cyan),
                "EpsilonBar".fg(Color::Cyan),
                "T6".fg(Color::Cyan),
                "K6".fg(Color::Cyan),
                "K6Bar".fg(Color::Cyan),
            ))
            .finish()
    }
}

#[derive(Debug)]
pub struct AmbiguousLorentzRank {
    ranks: Vec<i64>,
    spans: Vec<SimpleSpan>,
}

impl AmbiguousLorentzRank {
    pub(crate) fn new(ranks: Vec<i64>, spans: Vec<SimpleSpan>) -> Self {
        return Self { ranks, spans };
    }
}

impl Diagnostic for AmbiguousLorentzRank {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        let mut report = Report::build(ReportKind::Error, (name, self.spans[0].into_range()))
            .with_code(5)
            .with_message("Ambiguous rank in Lorentz expression");
        for (r, span) in self.ranks.iter().zip(self.spans.iter()) {
            report.add_label(
                Label::new((name, span.into_range()))
                    .with_message(format!("This expression has rank {}", r.fg(Color::Red)))
                    .with_color(Color::Red),
            );
        }
        return report.finish();
    }
}

#[derive(Debug)]
pub struct AmbiguousLorentzSpinMap {
    maps: Vec<Vec<i64>>,
    spans: Vec<SimpleSpan>,
}

impl AmbiguousLorentzSpinMap {
    pub(crate) fn new(maps: Vec<Vec<i64>>, spans: Vec<SimpleSpan>) -> Self {
        return Self { maps, spans };
    }
}

impl Diagnostic for AmbiguousLorentzSpinMap {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        let mut report = Report::build(ReportKind::Error, (name, self.spans[0].into_range()))
            .with_code(6)
            .with_message("Ambiguous spin map in Lorentz expression");
        for (m, span) in self.maps.iter().zip(self.spans.iter()) {
            let mut mappings = Vec::new();
            for (i, j) in m.iter().enumerate() {
                if (i as i64) < *j {
                    mappings.push(format!("{} ⟷ {}", i + 1, j + 1));
                }
            }
            report.add_label(
                Label::new((name, span.into_range()))
                    .with_message(format!(
                        "This expression connects the legs as {}",
                        (mappings.join(", ")).fg(Color::Red)
                    ))
                    .with_color(Color::Red),
            );
        }
        return report.finish();
    }
}

#[derive(Debug)]
pub struct UncontractedSpinIndex {
    index: i64,
    span: SimpleSpan,
}

impl UncontractedSpinIndex {
    pub(crate) fn new(index: i64, span: SimpleSpan) -> Self {
        return Self { index, span };
    }

    pub(crate) fn locate_span(s: &str, index: i64) -> SimpleSpan {
        let index_str = index_sub(index).chars().collect::<Vec<_>>();
        let mut k = 0;
        let mut span = 0..0;
        for (i, c) in s.chars().enumerate() {
            if c == index_str[k] {
                k += 1;
            }
            if k == index_str.len() {
                span = (i - k)..(i + 2);
                break;
            }
        }
        return span.into();
    }
}

impl Diagnostic for UncontractedSpinIndex {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        Report::build(ReportKind::Error, (name, self.span.into_range()))
            .with_code(7)
            .with_message(format!("Uncontracted spin index {}", self.index.fg(Color::Red)))
            .with_label(
                Label::new((name, self.span.into_range()))
                    .with_message(format!("Uncontracted spin index '{}'", self.index.fg(Color::Red)))
                    .with_color(Color::Red),
            )
            .with_help("Negative spin indices are internal and must appear exactly twice per product")
            .finish()
    }
}

// --------------------------------- Warnings ---------------------------------

#[derive(Debug)]
pub struct AmbiguousVertexRank {
    ranks: Vec<i64>,
    spans: Vec<SimpleSpan>,
}

impl AmbiguousVertexRank {
    pub(crate) fn new(ranks: Vec<i64>, spans: Vec<SimpleSpan>) -> Self {
        return Self { ranks, spans };
    }
}

impl Diagnostic for AmbiguousVertexRank {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        let mut report = Report::build(ReportKind::Warning, (name, self.spans[0].into_range()))
            .with_code(1)
            .with_message("Ambiguous rank in Lorentz expressions of the vertex");
        for (r, span) in self.ranks.iter().zip(self.spans.iter()) {
            report.add_label(
                Label::new((name, span.into_range()))
                    .with_message(format!("This expression has rank {}", r.fg(Color::Yellow)))
                    .with_color(Color::Yellow),
            );
        }
        return report.finish();
    }
}

#[derive(Debug)]
pub struct AmbiguousVertexSpinMap {
    maps: Vec<Vec<i64>>,
    spans: Vec<SimpleSpan>,
}

impl AmbiguousVertexSpinMap {
    pub(crate) fn new(maps: Vec<Vec<i64>>, spans: Vec<SimpleSpan>) -> Self {
        return Self { maps, spans };
    }
}

impl Diagnostic for AmbiguousVertexSpinMap {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        let mut report = Report::build(ReportKind::Warning, (name, self.spans[0].into_range()))
            .with_code(2)
            .with_message("Ambiguous spin map in Lorentz expressions of the vertex");
        for (m, span) in self.maps.iter().zip(self.spans.iter()) {
            let mut mappings = Vec::new();
            for (i, j) in m.iter().enumerate() {
                if (i as i64) < *j {
                    mappings.push(format!("{} ⟷ {}", i + 1, j + 1));
                }
            }
            report.add_label(
                Label::new((name, span.into_range()))
                    .with_message(format!(
                        "This expression connects the legs as {}",
                        (mappings.join(", ")).fg(Color::Yellow)
                    ))
                    .with_color(Color::Yellow),
            );
        }
        return report.finish();
    }
}

#[derive(Debug)]
pub struct AmbiguousCouplingOrder {
    coupling_orders: Vec<HashMap<String, usize>>,
    spans: Vec<SimpleSpan>,
}

impl AmbiguousCouplingOrder {
    pub(crate) fn new(coupling_orders: Vec<HashMap<String, usize>>, spans: Vec<SimpleSpan>) -> Self {
        return Self { coupling_orders, spans };
    }
}

impl Diagnostic for AmbiguousCouplingOrder {
    fn build<'a>(&self, name: &'a str) -> Report<'static, (&'a str, Range<usize>)> {
        let mut report = Report::build(ReportKind::Warning, (name, self.spans[0].into_range()))
            .with_code(2)
            .with_message("Ambiguous coupling order in vertex");
        for (map, span) in self.coupling_orders.iter().zip(self.spans.iter()) {
            report.add_label(
                Label::new((name, span.into_range()))
                    .with_message(format!(
                        "This expression has coupling orders {}",
                        format!("{:?}", map).fg(Color::Yellow)
                    ))
                    .with_color(Color::Yellow),
            );
        }
        return report.finish();
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "emit-test-reports")]
    use crate::parser::diagnostic::{AmbiguousVertexRank, AmbiguousVertexSpinMap, Diagnostic, InvalidIntDigit};

    #[cfg(feature = "emit-test-reports")]
    #[test]
    fn diagnostic_test() {
        let source = "134u12";
        let name = "diagnostic_test";
        let d = InvalidIntDigit::new('u', (3..4).into());
        d.emit(name, source).unwrap();
    }

    #[cfg(feature = "emit-test-reports")]
    #[test]
    fn diagnostic_warn_test() {
        let expr = "P(1, -1) * P(1, -1) * Gamma5(1,1,2) + Psq(1, -1) * P(1, -1)";
        let name = "lorentz_split_test";
        let d = AmbiguousVertexRank::new(vec![2, 3], vec![(0..35).into(), (38..59).into()]);
        d.emit(name, expr).unwrap();
    }

    #[cfg(feature = "emit-test-reports")]
    #[test]
    fn diagnostic_spin_warn_test() {
        let expr = "IdentityL(1, 2) * IdentityL(3, 4) + IdentityL(1, 4) * Identity(3, 2)";
        let name = "lorentz_spin_test";
        let d = AmbiguousVertexSpinMap::new(
            vec![vec![2, 1, 4, 3], vec![4, 3, 2, 1]],
            vec![(0..33).into(), (36..68).into()],
        );
        d.emit(name, expr).unwrap();
    }
}
