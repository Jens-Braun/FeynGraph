use std::collections::HashMap;
use std::path::Path;
use itertools::Itertools;
use thiserror::Error;
use crate::model::ufoparser::{Rule, UFOParser};

mod ufoparser;

#[allow(clippy::large_enum_variant)]
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Encountered illegal model option '{0}'")]
    ContentError(String),
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error("Error while parsing UFO model")]
    ParseError(#[from] pest::error::Error<Rule>),
}

#[derive(PartialEq, Debug, Hash, Clone)]
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

#[derive(Debug, PartialEq, Hash, Clone)]
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

#[derive(Debug, PartialEq, Clone)]
pub struct Vertex {
    name: String,
    particles: Vec<String>,
    couplings_orders: HashMap<String, usize>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Model {
    particles: HashMap<String, Particle>,
    vertices: HashMap<String, Vertex>,
    coupling_orders: Vec<String>
}

impl Model {
    pub fn from_ufo(path: &Path) -> Result<Self, ModelError> {
        return UFOParser::parse_ufo_model(path);
    }
}

#[derive(Clone, PartialEq, Debug)]
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
            vertex_degrees: vertex_degrees.into_iter().sorted().dedup().collect_vec(),
        }
    }
}

impl From<Model> for TopologyModel {
    fn from(model: Model) -> Self {
        let mut vertex_degrees = Vec::new();
        for (_, vertex) in model.vertices.iter() {
            vertex_degrees.push(vertex.particles.len());
        }
        return Self {
            vertex_degrees: vertex_degrees.into_iter().sorted().dedup().collect_vec(),
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use crate::model::{Model, TopologyModel};

    #[test]
    fn model_conversion_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/Standard_Model_UFO")).unwrap();
        let topology_model = TopologyModel::from(&model);
        assert_eq!(topology_model, TopologyModel {vertex_degrees: vec![3, 4]});
    }
}