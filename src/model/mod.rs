use std::collections::HashMap;
use std::path::PathBuf;
use itertools::Itertools;
use pyo3::exceptions::{PyIOError, PySyntaxError};
use thiserror::Error;
use crate::model::ufoparser::{Rule, UFOParser};

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

mod ufoparser;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Encountered illegal model option '{0}'")]
    ContentError(String),
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error("Error while parsing UFO model")]
    ParseError(#[from] pest::error::Error<Rule>),
}

#[cfg(feature = "python-bindings")]
impl std::convert::From<ModelError> for PyErr {
    fn from(err: ModelError) -> PyErr {
        match err {
            ModelError::IOError(_) => PyIOError::new_err(err.to_string()),
            ModelError::ParseError(_) => PySyntaxError::new_err(err.to_string()),
            ModelError::ContentError(_) => PySyntaxError::new_err(err.to_string()),
        }
    }
}

#[derive(PartialEq, Debug, Hash)]
pub enum LineStyle {
    Dashed,
    Dotted,
    Straight,
    Wavy,
    Curly,
    Scurly,
    Swavy,
    Double
}

#[derive(Debug, PartialEq, Hash)]
pub struct Particle {
    name: String,
    pdg_code: isize,
    texname: String,
    antitexname: String,
    linestyle: LineStyle,
}

impl Particle {
    pub fn new(name: impl Into<String>,
               pdg_code: isize,
               texname: impl Into<String>,
               antitexname: impl Into<String>,
               linestyle: LineStyle
    ) -> Self {
        return Self {
            name: name.into(),
            pdg_code,
            texname: texname.into(),
            antitexname: antitexname.into(),
            linestyle,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Vertex {
    name: String,
    particles: Vec<String>,
    couplings_orders: HashMap<String, usize>,
}

#[cfg_attr(feature = "python-bindings", pyclass)]
#[derive(Debug, PartialEq)]
pub struct Model {
    particles: HashMap<String, Particle>,
    vertices: HashMap<String, Vertex>,
    coupling_orders: Vec<String>
}

#[cfg_attr(feature = "python-bindings", pymethods)]
impl Model {
    
    #[cfg(feature = "python-bindings")]
    #[pyo3(name = "from_ufo")]
    #[staticmethod]
    fn from_ufo_py(path: PathBuf) -> PyResult<Self> {
        return Ok(UFOParser::parse_ufo_model(&path)?);
    }
}

#[cfg_attr(feature = "python-bindings", pyclass)]
#[derive(Clone)]
pub struct TopologyModel {
    vertex_degrees: Vec<usize>
}

impl From<&Model> for TopologyModel {
    fn from(model: &Model) -> Self {
        let mut vertex_degrees = Vec::new();
        for (_, vertex) in model.vertices.iter() {
            vertex_degrees.push(vertex.particles.len());
        }
        return Self {
            vertex_degrees: vertex_degrees.into_iter().dedup().sorted().collect_vec(),
        }
    }
}

impl From<Vec<usize>> for TopologyModel {
    fn from(vec: Vec<usize>) -> Self {
        return Self {
            vertex_degrees: vec
        }
    }
}

impl TopologyModel {
    pub fn get(&self, i: usize) -> usize {
        return self.vertex_degrees[i];
    }
    
    pub fn degrees_iter(&self) -> impl Iterator<Item=usize> {
        return self.vertex_degrees.clone().into_iter();
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl TopologyModel {
    #[new]
    fn new(degrees: Vec<usize>) -> Self {
        return TopologyModel { vertex_degrees: degrees };
    }
}