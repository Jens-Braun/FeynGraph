use crate::model::Statistic::Fermi;
use indexmap::IndexMap;
use itertools::Itertools;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

mod qgraf_parser;
mod ufo_parser;

#[allow(clippy::large_enum_variant)]
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Encountered illegal model option: {0}")]
    ContentError(String),
    #[error("Error wile trying to access file {0}: {1}")]
    IOError(String, #[source] std::io::Error),
    #[error("Error while parsing file {0}: {1}")]
    ParseError(String, #[source] peg::error::ParseError<peg::str::LineCol>),
}

#[derive(PartialEq, Debug, Hash, Clone, Eq)]
pub enum LineStyle {
    Dashed,
    Dotted,
    Straight,
    Wavy,
    Curly,
    Scurly,
    Swavy,
    Double,
}

#[derive(PartialEq, Debug, Hash, Clone, Eq)]
pub enum Statistic {
    Fermi,
    Bose,
}

#[derive(Debug, PartialEq, Hash, Clone, Eq)]
pub struct Particle {
    name: String,
    anti_name: String,
    pdg_code: isize,
    texname: String,
    antitexname: String,
    linestyle: LineStyle,
    self_anti: bool,
    statistic: Statistic,
}

impl Particle {
    pub fn get_name(&self) -> &String {
        return &self.name;
    }

    pub fn get_anti_name(&self) -> &String {
        return &self.anti_name;
    }

    pub fn get_pdg(&self) -> isize {
        return self.pdg_code;
    }

    pub fn is_anti(&self) -> bool {
        return self.pdg_code <= 0;
    }

    pub fn into_anti(self) -> Particle {
        return Self {
            name: self.anti_name,
            anti_name: self.name,
            pdg_code: -self.pdg_code,
            texname: self.antitexname,
            antitexname: self.texname,
            linestyle: self.linestyle,
            self_anti: self.self_anti,
            statistic: self.statistic,
        };
    }
}

impl Particle {
    pub fn new(
        name: impl Into<String>,
        anti_name: impl Into<String>,
        pdg_code: isize,
        texname: impl Into<String>,
        antitexname: impl Into<String>,
        linestyle: LineStyle,
        statistic: Statistic,
    ) -> Self {
        let texname = texname.into();
        let antitexname = antitexname.into();
        let self_anti = texname == antitexname;
        return Self {
            name: name.into(),
            anti_name: anti_name.into(),
            pdg_code,
            texname,
            antitexname,
            linestyle,
            self_anti,
            statistic,
        };
    }

    pub fn self_anti(&self) -> bool {
        return self.self_anti;
    }

    pub fn is_fermi(&self) -> bool {
        return self.statistic == Fermi;
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct InteractionVertex {
    pub(crate) name: String,
    pub(crate) particles: Vec<String>,
    pub(crate) spin_map: Vec<isize>,
    pub(crate) coupling_orders: HashMap<String, usize>,
}

impl InteractionVertex {
    pub fn particles_iter(&self) -> impl Iterator<Item = &String> {
        return self.particles.iter();
    }

    pub fn count_particles(&self, key: &String) -> usize {
        return self.particles.iter().filter(|k| **k == *key).count();
    }

    pub fn get_coupling_orders(&self) -> &HashMap<String, usize> {
        return &self.coupling_orders;
    }

    pub fn get_degree(&self) -> usize {
        return self.particles.len();
    }
}

impl std::fmt::Display for InteractionVertex {
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
    vertices: IndexMap<String, InteractionVertex>,
    couplings: Vec<String>,
}

impl Default for Model {
    fn default() -> Self {
        return ufo_parser::sm();
    }
}

impl Model {
    pub fn from_ufo(path: &Path) -> Result<Self, ModelError> {
        return ufo_parser::parse_ufo_model(path);
    }

    pub fn from_qgraf(path: &Path) -> Result<Self, ModelError> {
        return qgraf_parser::parse_qgraf_model(path);
    }

    pub fn get_anti_index(&self, particle_index: usize) -> usize {
        return if self.particles[particle_index].self_anti {
            particle_index
        } else {
            self.particles
                .values()
                .find_position(|p| p.pdg_code == -self.particles[particle_index].pdg_code)
                .as_ref()
                .unwrap()
                .0
        };
    }

    pub fn get_anti(&self, particle_index: usize) -> &Particle {
        return if self.particles[particle_index].self_anti {
            &self.particles[particle_index]
        } else {
            self.particles
                .values()
                .find(|p| p.pdg_code == -self.particles[particle_index].pdg_code)
                .as_ref()
                .unwrap()
        };
    }

    pub fn normalize(&self, particle_index: usize) -> usize {
        return if self.particles[particle_index].pdg_code < 0 {
            self.get_anti_index(particle_index)
        } else {
            particle_index
        };
    }

    pub fn get_particle(&self, index: usize) -> &Particle {
        return &self.particles[index];
    }

    pub fn get_particle_name(&self, name: &str) -> Option<&Particle> {
        return self.particles.get(name);
    }

    pub fn find_particle(&self, name: &str) -> Result<&Particle, ModelError> {
        return if let Some(particle) = self.particles.get(name) {
            Ok(particle)
        } else {
            Err(ModelError::ContentError(format!(
                "Particle '{}' not found in model",
                name
            )))
        };
    }

    pub fn get_particle_index(&self, key: &str) -> Result<usize, ModelError> {
        return self
            .particles
            .get_index_of(key)
            .ok_or_else(|| ModelError::ContentError(format!("Particle '{}' not found in model", key)));
    }

    pub fn vertex(&self, index: usize) -> &InteractionVertex {
        return &self.vertices[index];
    }

    pub fn vertices_iter(&self) -> impl Iterator<Item = &InteractionVertex> {
        return self.vertices.values();
    }

    pub fn particles_iter(&self) -> impl Iterator<Item = &Particle> {
        return self.particles.values();
    }

    pub fn n_vertices(&self) -> usize {
        return self.vertices.len();
    }

    pub fn coupling_orders(&self) -> &Vec<String> {
        return &self.couplings;
    }

    /// Check if adding `vertex` to the diagram is allowed by the maximum power of the coupling constants
    pub(crate) fn check_coupling_orders(
        &self,
        interaction: usize,
        remaining_coupling_orders: &Option<HashMap<String, usize>>,
    ) -> bool {
        return if let Some(ref remaining_orders) = remaining_coupling_orders {
            for (coupling, order) in self.vertices[interaction].get_coupling_orders() {
                if let Some(remaining_order) = remaining_orders.get(coupling) {
                    if order > remaining_order {
                        return false;
                    }
                } else {
                    continue;
                }
            }
            true
        } else {
            true
        };
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct TopologyModel {
    vertex_degrees: Vec<usize>,
}

impl From<&Model> for TopologyModel {
    fn from(model: &Model) -> Self {
        let mut vertex_degrees = Vec::new();
        for (_, vertex) in model.vertices.iter() {
            vertex_degrees.push(vertex.particles.len());
        }
        return Self {
            vertex_degrees: vertex_degrees.into_iter().sorted().dedup().collect_vec(),
        };
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
        };
    }
}

impl From<Vec<usize>> for TopologyModel {
    fn from(vec: Vec<usize>) -> Self {
        return Self { vertex_degrees: vec };
    }
}

impl TopologyModel {
    pub fn get(&self, i: usize) -> usize {
        return self.vertex_degrees[i];
    }

    pub fn degrees_iter(&self) -> impl Iterator<Item = usize> {
        return self.vertex_degrees.clone().into_iter();
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{Model, TopologyModel};
    use std::path::PathBuf;
    use test_log::test;

    #[test]
    fn model_conversion_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/Standard_Model_UFO")).unwrap();
        let topology_model = TopologyModel::from(&model);
        assert_eq!(
            topology_model,
            TopologyModel {
                vertex_degrees: vec![3, 4]
            }
        );
    }
}
