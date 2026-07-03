use drawing::{Color, Decoration, DecorationKind, PathStyle, Stroke, Theme};
use pyo3::prelude::*;

#[pyclass(name = "Theme")]
pub(crate) struct PyTheme(Theme);

#[pymethods]
impl PyTheme {
    #[new]
    fn default() -> Self {
        Self(Theme::default())
    }

    fn set_global(&self) {
        Theme::set_global(self.0.clone());
    }

    fn set_particle(&mut self, p: String, style: PyPathStyle) {
        self.0.set_particle(p, style.0);
    }

    fn set_vertex(&mut self, v: String, style: PyDecoration) {
        self.0.set_vertex(v, style.0);
    }
}

#[pyclass(name = "PathStyle", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyPathStyle(PathStyle);

#[pymethods]
impl PyPathStyle {
    #[new]
    #[pyo3(signature = (stroke = None, width = None, color = None, decoration = None, n_segments = None))]
    fn __new__(
        stroke: Option<PyStroke>,
        width: Option<f64>,
        color: Option<PyColor>,
        decoration: Option<PyDecoration>,
        n_segments: Option<usize>,
    ) -> Self {
        Self(PathStyle {
            stroke: stroke.map(|s| s.into()).unwrap_or(Stroke::default()),
            width: width.unwrap_or(0.006),
            color: color.map(|c| c.0).unwrap_or(Color::default()),
            decoration: decoration.map(|d| d.0),
            n_segments: n_segments.unwrap_or(Theme::get_global().pattern_count),
            ..Default::default()
        })
    }

    #[getter]
    fn get_stroke(&self) -> PyStroke {
        self.0.stroke.clone().into()
    }
    #[setter]
    fn set_stroke(&mut self, stroke: PyStroke) {
        self.0.stroke = stroke.into();
    }
    #[getter]
    fn get_width(&self) -> f64 {
        self.0.width
    }
    #[setter]
    fn set_width(&mut self, width: f64) {
        self.0.width = width;
    }
    #[getter]
    fn get_color(&self) -> PyColor {
        PyColor(self.0.color)
    }
    #[setter]
    fn set_color(&mut self, color: PyColor) {
        self.0.color = color.0;
    }
    #[getter]
    fn get_decoration(&self) -> Option<PyDecoration> {
        self.0.decoration.clone().map(|d| PyDecoration(d))
    }
    #[setter]
    fn set_decoration(&mut self, decoration: PyDecoration) {
        self.0.decoration = Some(decoration.0);
    }
    #[getter]
    fn get_segments(&self) -> usize {
        self.0.n_segments
    }
    #[setter]
    fn set_segments(&mut self, n_segments: usize) {
        self.0.n_segments = n_segments;
    }
}

#[pyclass(name = "Stroke", from_py_object)]
#[derive(Default, Debug, Clone)]
pub(crate) enum PyStroke {
    #[default]
    Solid,
    Dashed,
    Dotted,
}

impl From<PyStroke> for Stroke {
    fn from(value: PyStroke) -> Self {
        match value {
            PyStroke::Dashed => Stroke::Dashed,
            PyStroke::Dotted => Stroke::Dotted,
            PyStroke::Solid => Stroke::Solid,
        }
    }
}

impl From<Stroke> for PyStroke {
    fn from(value: Stroke) -> Self {
        match value {
            Stroke::Dashed => PyStroke::Dashed,
            Stroke::Dotted => PyStroke::Dotted,
            Stroke::Solid => PyStroke::Solid,
        }
    }
}

#[pymethods]
impl PyStroke {
    #[new]
    fn __new__() -> Self {
        Self::default()
    }
}

#[pyclass(name = "Decoration", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyDecoration(Decoration);

#[pymethods]
impl PyDecoration {
    #[new]
    #[pyo3(signature = (kind, pos = None, angle = None, size = None, color = None))]
    fn __new__(
        kind: PyDecorationKind,
        pos: Option<f64>,
        angle: Option<f64>,
        size: Option<f64>,
        color: Option<PyColor>,
    ) -> Self {
        Self(Decoration {
            pos: pos.unwrap_or(0.5),
            rotate: angle.unwrap_or(0.),
            size: size.unwrap_or(Theme::get_global().decoration_size),
            color: color.map(|c| c.0).unwrap_or(Color::default()),
            kind: kind.into(),
        })
    }

    #[getter]
    fn get_pos(&self) -> PyResult<f64> {
        Ok(self.0.pos)
    }
    #[setter]
    fn set_pos(&mut self, pos: f64) {
        self.0.pos = pos;
    }
    #[getter]
    fn get_angle(&self) -> PyResult<f64> {
        Ok(self.0.rotate)
    }
    #[setter]
    fn set_angle(&mut self, angle: f64) {
        self.0.rotate = angle;
    }
    #[getter]
    fn get_color(&self) -> PyResult<PyColor> {
        Ok(PyColor(self.0.color))
    }
    #[setter]
    fn set_color(&mut self, color: PyColor) {
        self.0.color = color.0;
    }
    #[getter]
    fn get_size(&self) -> PyResult<f64> {
        Ok(self.0.size)
    }
    #[setter]
    fn set_size(&mut self, size: f64) {
        self.0.size = size;
    }
    #[getter]
    fn get_kind(&self) -> PyResult<PyDecorationKind> {
        Ok(self.0.kind.clone().into())
    }
    #[setter]
    fn set_kind(&mut self, kind: PyDecorationKind) {
        self.0.kind = kind.into();
    }
}

#[pyclass(name = "DecorationKind", from_py_object)]
#[derive(Clone, Debug)]
pub(crate) enum PyDecorationKind {
    Cross,
    Triangle,
    Box,
    Circle,
}

impl From<PyDecorationKind> for DecorationKind {
    fn from(value: PyDecorationKind) -> Self {
        match value {
            PyDecorationKind::Box => DecorationKind::Box,
            PyDecorationKind::Circle => DecorationKind::Circle,
            PyDecorationKind::Cross => DecorationKind::Cross,
            PyDecorationKind::Triangle => DecorationKind::Triangle,
        }
    }
}

impl From<DecorationKind> for PyDecorationKind {
    fn from(value: DecorationKind) -> Self {
        match value {
            DecorationKind::Box => PyDecorationKind::Box,
            DecorationKind::Circle => PyDecorationKind::Circle,
            DecorationKind::Cross => PyDecorationKind::Cross,
            DecorationKind::Triangle => PyDecorationKind::Triangle,
        }
    }
}

#[pyclass(name = "Color", from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyColor(Color);

#[pymethods]
impl PyColor {
    #[staticmethod]
    fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self(Color { r, g, b })
    }

    #[staticmethod]
    fn black() -> Self {
        Self::rgb(0, 0, 0)
    }
    #[staticmethod]
    fn dark_gray() -> Self {
        Self::rgb(96, 96, 96)
    }
    #[staticmethod]
    fn gray() -> Self {
        Self::rgb(160, 160, 160)
    }
    #[staticmethod]
    fn light_gray() -> Self {
        Self::rgb(220, 220, 220)
    }
    #[staticmethod]
    fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    #[staticmethod]
    fn brown() -> Self {
        Self::rgb(165, 42, 42)
    }
    #[staticmethod]
    fn dark_red() -> Self {
        Self::rgb(0x8B, 0, 0)
    }
    #[staticmethod]
    fn red() -> Self {
        Self::rgb(255, 0, 0)
    }
    #[staticmethod]
    fn light_red() -> Self {
        Self::rgb(255, 128, 128)
    }

    #[staticmethod]
    fn cyan() -> Self {
        Self::rgb(0, 255, 255)
    }
    #[staticmethod]
    fn magenta() -> Self {
        Self::rgb(255, 0, 255)
    }
    #[staticmethod]
    fn yellow() -> Self {
        Self::rgb(255, 255, 0)
    }

    #[staticmethod]
    fn orange() -> Self {
        Self::rgb(255, 165, 0)
    }
    #[staticmethod]
    fn light_yellow() -> Self {
        Self::rgb(255, 255, 0xE0)
    }
    #[staticmethod]
    fn khaki() -> Self {
        Self::rgb(240, 230, 140)
    }

    #[staticmethod]
    fn dark_green() -> Self {
        Self::rgb(0, 0x64, 0)
    }
    #[staticmethod]
    fn green() -> Self {
        Self::rgb(0, 255, 0)
    }
    #[staticmethod]
    fn light_green() -> Self {
        Self::rgb(0x90, 0xEE, 0x90)
    }

    #[staticmethod]
    fn dark_blue() -> Self {
        Self::rgb(0, 0, 0x8B)
    }
    #[staticmethod]
    fn blue() -> Self {
        Self::rgb(0, 0, 255)
    }
    #[staticmethod]
    fn light_blue() -> Self {
        Self::rgb(0xAD, 0xD8, 0xE6)
    }
}
