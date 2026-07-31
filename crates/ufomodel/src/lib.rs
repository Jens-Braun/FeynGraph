use model::{ModelBase, ModelError};
use std::path::Path;
use util::{HashMap, IndexMap};

use components::*;

#[cfg(feature = "py-reader")]
use crate::python::cpython_reader::CPythonReader;
use crate::{error::ReadError, parser::UFOReader};

pub mod components;
pub mod error;
pub mod parser;
#[cfg(feature = "py-reader")]
pub mod python;
#[cfg(feature = "serde")]
pub(crate) mod serde_util;

pub enum Reader {
    Native,
    #[cfg(feature = "py-reader")]
    CPython,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(PartialEq, Debug)]
pub struct UFOModel {
    pub particles: IndexMap<String, Particle>,
    pub vertices: IndexMap<String, Vertex>,
    pub couplings: HashMap<String, Coupling>,
    pub lorentz: HashMap<String, Lorentz>,
    pub parameters: HashMap<String, Parameter>,
    pub orders: HashMap<String, CouplingOrder>,
    pub ct_parameters: HashMap<String, CTParameter>,
    pub ct_vertices: IndexMap<String, CTVertex>,

    pub anti_map: Vec<usize>,
}

impl UFOModel {
    pub fn read(path: &Path, reader: Reader) -> Result<UFOModel, ReadError> {
        let mut ufo_reader = UFOReader::new(path);
        match reader {
            Reader::Native => todo!(),
            #[cfg(feature = "py-reader")]
            Reader::CPython => ufo_reader.read_with::<CPythonReader>(),
        }
    }
}

impl ModelBase for UFOModel {
    type Particle = Particle;
    type Vertex = Vertex;

    fn particle(&self, index: usize) -> &Self::Particle {
        &self.particles[index]
    }

    fn particle_by_name(&self, name: impl AsRef<str>) -> Result<&Self::Particle, model::ModelError> {
        return self
            .particles
            .get(name.as_ref())
            .ok_or_else(|| ModelError::particle_not_found(name.as_ref()));
    }

    fn particle_index_by_name(&self, name: impl AsRef<str>) -> Result<usize, model::ModelError> {
        return self
            .particles
            .get_index_of(name.as_ref())
            .ok_or_else(|| ModelError::particle_not_found(name.as_ref()));
    }

    fn anti_particle(&self, particle: usize) -> &Self::Particle {
        &self.particles[self.anti_map[particle]]
    }

    fn anti_particle_index(&self, particle: usize) -> usize {
        self.anti_map[particle]
    }

    fn vertex(&self, index: usize) -> &Self::Vertex {
        &self.vertices[index]
    }

    fn particles(&self) -> impl ExactSizeIterator<Item = &Self::Particle> {
        self.particles.values()
    }

    fn vertices(&self) -> impl ExactSizeIterator<Item = &Self::Vertex> {
        self.vertices.values()
    }

    fn couplings(&self) -> impl ExactSizeIterator<Item = impl AsRef<str>> {
        self.orders.values().map(|c| &c.name)
    }

    fn supports_sign(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "py-reader", feature = "serde"))]
    use std::path::PathBuf;

    #[cfg(all(feature = "py-reader", feature = "serde"))]
    use super::*;

    #[test]
    #[cfg(all(feature = "py-reader", feature = "serde"))]
    fn serde_test() {
        let model = UFOModel::read(&PathBuf::from("tests/resources/UFO_QL_all_operators"), Reader::CPython).unwrap();
        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: UFOModel = serde_json::from_str(&serialized).unwrap();
        assert_eq!(model, deserialized);
    }
}
