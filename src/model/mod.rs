use std::collections::HashMap;
use std::path::Path;
use itertools::Itertools;
use thiserror::Error;
use indexmap::IndexMap;
use crate::model::ufoparser::{Rule, UFOParser};
use crate::topology::Topology;

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
    self_anti: bool,
}

impl Particle {
    pub fn get_name(&self) -> &String {
        return &self.name;
    }
    
    pub fn is_anti(&self) -> bool {
        return self.pdg_code <= 0;
    }
    
    pub fn into_anti(self, anti_name: String) -> Particle {
        return Self {
            name: anti_name,
            pdg_code: -self.pdg_code,
            texname: self.antitexname,
            antitexname: self.texname,
            linestyle: self.linestyle,
            self_anti: self.self_anti,
        }
    }
}

impl Particle {
    pub fn new(name: impl Into<String>,
               pdg_code: isize,
               texname: impl Into<String>,
               antitexname: impl Into<String>,
               linestyle: LineStyle
    ) -> Self {
        let texname = texname.into();
        let antitexname = antitexname.into();
        let self_anti = texname == antitexname;
        return Self {
            name: name.into(),
            pdg_code,
            texname,
            antitexname,
            linestyle,
            self_anti
        }
    }
    
    pub fn self_anti(&self) -> bool {
        return self.self_anti;
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Vertex {
    name: String,
    particles: Vec<String>,
    couplings_orders: HashMap<String, usize>,
}

impl Vertex {
    pub fn particles_iter(&self) -> impl Iterator<Item = &String> {
        return self.particles.iter();
    } 
    
    pub fn count_particles(&self, key: &String) -> usize {
        return self.particles.iter().filter(|k| **k == *key).count();
    }
    
    pub fn get_coupling_orders(&self) -> &HashMap<String, usize> {
        return &self.couplings_orders;
    }
    
    pub fn get_degree(&self) -> usize {
        return self.particles.len();
    }
}

impl std::fmt::Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[ ", self.name)?;
        for p in self.particles.iter() {
            write!(f, "{} ", p)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Model {
    particles: IndexMap<String, Particle>,
    vertices: IndexMap<String, Vertex>,
    coupling_orders: Vec<String>
}

impl Model {
    pub fn from_ufo(path: &Path) -> Result<Self, ModelError> {
        return UFOParser::parse_ufo_model(path);
    }
    
    pub fn get_anti(&self, particle_index: usize) -> usize {
        return if self.particles[particle_index].self_anti {
            particle_index
        } else {
            self.particles.values().find_position(
                |p| p.pdg_code == -self.particles[particle_index].pdg_code
            ).as_ref().unwrap().0
        }
    }

    pub fn normalize(&self, particle_index: usize) -> usize {
        if self.particles[particle_index].pdg_code < 0 {
            return self.get_anti(particle_index);
        } else {
            return particle_index;
        }
    }

    pub fn get_particle(&self, index: usize) -> &Particle {
        return &self.particles[index];
    }
    
    pub fn get_particle_name(&self, name: &str) -> Option<&Particle> {
        return self.particles.get(name);
    }
    
    pub fn get_particle_index(&self, key: &String) -> usize {
        return self.particles.get_index_of(key).unwrap();
    }
    
    pub fn get_vertex(&self, index: usize) -> &Vertex {
        return &self.vertices[index];
    }
    
    pub fn vertices_iter(&self) -> impl Iterator<Item = &Vertex> {
        return self.vertices.values();
    }

    /// Check if adding `vertex` to the diagram is allowed by the maximum power of the coupling constants
    pub(crate) fn check_coupling_orders(&self, interaction: usize, 
                             remaining_coupling_orders: &Option<HashMap<String, usize>>) -> bool {
        return if let Some(ref remaining_orders) = remaining_coupling_orders {
            for (coupling, order) in self.vertices[interaction].get_coupling_orders() {
                if let Some(remaining_order) = remaining_orders.get(coupling) {
                    if order > remaining_order { return false; }
                } else { continue; }
            }
            true
        } else { true }
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