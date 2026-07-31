use util::parser::Diagnostic;

#[derive(Debug)]
pub struct ReadError {
    message: String,
    kind: ReadErrorKind,
}

impl ReadError {
    pub(crate) fn new(kind: ReadErrorKind, message: impl Into<String>) -> Self {
        return Self {
            message: message.into(),
            kind,
        };
    }

    pub fn kind(&self) -> ReadErrorKind {
        return self.kind;
    }
}

impl std::error::Error for ReadError {}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum ReadErrorKind {
    IOError(std::io::ErrorKind),
    NulError,
    PyError,
    ParseError,
    CastError,
    InvalidModelError,
}

impl From<std::io::Error> for ReadError {
    fn from(value: std::io::Error) -> Self {
        return Self::new(ReadErrorKind::IOError(value.kind()), format!("IOError: {value}"));
    }
}

impl From<std::ffi::NulError> for ReadError {
    fn from(_value: std::ffi::NulError) -> Self {
        return Self::new(ReadErrorKind::NulError, "Unexpected nul bytes");
    }
}

#[cfg(feature = "py-reader")]
impl From<pyo3::PyErr> for ReadError {
    fn from(value: pyo3::PyErr) -> Self {
        return Self::new(ReadErrorKind::PyError, value.to_string());
    }
}

#[cfg(feature = "py-reader")]
impl From<pyo3::CastIntoError<'_>> for ReadError {
    fn from(value: pyo3::CastIntoError) -> Self {
        return Self::new(ReadErrorKind::CastError, value.to_string());
    }
}

#[derive(Debug)]
pub struct ParseError {
    name: String,
    input: String,
    diagnostics: Vec<Box<dyn Diagnostic>>,
}

impl ParseError {
    pub(crate) fn new(
        name: impl Into<String>,
        input: impl Into<String>,
        diagnostics: Vec<Box<dyn Diagnostic>>,
    ) -> Self {
        Self {
            name: name.into(),
            input: input.into(),
            diagnostics,
        }
    }

    pub(crate) fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    pub(crate) fn into_diagnostics(self) -> Vec<Box<dyn Diagnostic>> {
        return self.diagnostics;
    }

    pub fn emit(&self) -> std::io::Result<()> {
        let src = ariadne::Source::from(self.input.as_str());
        for err in &self.diagnostics {
            err.emit_cached(&self.name, &src)?;
        }
        Ok(())
    }
}
