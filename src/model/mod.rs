//! A physical model used for diagram generation and drawing.

use crate::util::{HashMap, IndexMap};
use itertools::Itertools;
use std::path::Path;
use thiserror::Error;

mod qgraf_parser;
mod ufo_parser;

/// Custom error type for errors specific to a model.
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

/// Line style of a propagator, specified by the UFO 2.0 standard.
///
/// This property is used for drawing propagators.
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
    None,
}

/// Statistic deciding the commutation property of a field.
#[derive(PartialEq, Debug, Hash, Clone, Eq)]
pub enum Statistic {
    Fermi,
    Bose,
}

/// Internal representation of a particle.
///
/// Contains only the information necessary for diagram generation and drawing.
#[derive(Debug, PartialEq, Hash, Clone, Eq)]
pub struct Particle {
    pub(crate) name: String,
    pub(crate) anti_name: String,
    pub(crate) pdg_code: isize,
    pub(crate) texname: String,
    pub(crate) antitexname: String,
    pub(crate) linestyle: LineStyle,
    pub(crate) self_anti: bool,
    pub(crate) statistic: Statistic,
}

impl Particle {
    /// Get the particle's name. Corresponds to the UFO property `name`.
    pub fn name(&self) -> &String {
        return &self.name;
    }

    /// Get the name of the particle's anti-particle. Corresponds the the UFO property `antiname`.
    pub fn anti_name(&self) -> &String {
        return &self.anti_name;
    }

    /// Get the particle's PDG ID. Corresponds the the UFO property `pdg_code`.
    pub fn pdg(&self) -> isize {
        return self.pdg_code;
    }

    /// Query whether the particle is an anti-particle, decided by the sign of the PDG ID.
    pub fn is_anti(&self) -> bool {
        return self.pdg_code <= 0;
    }

    /// Query whether the particle is its own anti-particle.
    pub fn self_anti(&self) -> bool {
        return self.self_anti;
    }

    /// Query whether the particle obeys Fermi-Dirac statistics.
    pub fn is_fermi(&self) -> bool {
        return self.statistic == Statistic::Fermi;
    }

    pub(crate) fn into_anti(self) -> Particle {
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

    pub(crate) fn new(
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
}

/// Internal representation of an interaction vertex.
///
/// Contains only the information necessary for diagram generation and drawing.
#[derive(Debug, PartialEq, Clone)]
pub struct InteractionVertex {
    pub(crate) name: String,
    pub(crate) particles: Vec<String>,
    pub(crate) spin_map: Vec<isize>,
    pub(crate) coupling_orders: HashMap<String, usize>,
    pub(crate) particle_counts: HashMap<usize, usize>,
}

impl InteractionVertex {
    /// Get an iterator over the names of the particles attached to this vertex.
    pub fn particles_iter(&self) -> impl Iterator<Item = &String> {
        return self.particles.iter();
    }

    /// Get a map of the powers of couplings of the vertex.
    pub fn coupling_orders(&self) -> &HashMap<String, usize> {
        return &self.coupling_orders;
    }

    /// Get the degree of the vertex, i.e. the number of particles attached to it.
    pub fn degree(&self) -> usize {
        return self.particles.len();
    }

    pub(crate) fn new(
        name: String,
        particles: Vec<String>,
        spin_map: Vec<isize>,
        coupling_orders: HashMap<String, usize>,
    ) -> Self {
        return Self {
            name,
            particles,
            spin_map,
            coupling_orders,
            particle_counts: HashMap::default(),
        };
    }

    pub(crate) fn build_counts(&mut self, particles: &IndexMap<String, Particle>) {
        for (p, count) in self.particles.iter().counts() {
            self.particle_counts.insert(particles.get_index_of(p).unwrap(), count);
        }
    }

    pub(crate) fn particle_counts(&self) -> &HashMap<usize, usize> {
        return &self.particle_counts;
    }

    pub(crate) fn particle_count(&self, particle: &usize) -> usize {
        if let Some(c) = self.particle_counts.get(particle) {
            return *c;
        } else {
            return 0;
        }
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

/// Internal representation of a physical model.
///
/// This model structure strongly resembles the Python representation of a UFO model, but only contains the information
/// necessary for diagram generation and drawing.
///
/// A model can currently be imported from two formats, [UFO 2.0](https://arxiv.org/abs/2304.09883)
/// models and [QGRAF](http://cefema-gt.tecnico.ulisboa.pt/~paulo/qgraf.html) models.
///
/// The [`default`](Self::default()) implementation of the model is the Standard Model in Feynman gauge.
#[derive(Debug, PartialEq, Clone)]
pub struct Model {
    particles: IndexMap<String, Particle>,
    vertices: IndexMap<String, InteractionVertex>,
    couplings: Vec<String>,
    anti_map: Vec<usize>,
}

impl Default for Model {
    fn default() -> Self {
        return ufo_parser::sm();
    }
}

impl Model {
    pub(crate) fn new(
        particles: IndexMap<String, Particle>,
        mut vertices: IndexMap<String, InteractionVertex>,
        couplings: Vec<String>,
    ) -> Self {
        let anti_map = particles
            .values()
            .enumerate()
            .map(|(i, p)| {
                if p.self_anti {
                    i
                } else {
                    particles
                        .values()
                        .find_position(|q| q.pdg_code == -p.pdg_code)
                        .as_ref()
                        .unwrap()
                        .0
                }
            })
            .collect_vec();
        for v in vertices.values_mut() {
            v.build_counts(&particles);
        }
        return Self {
            particles,
            vertices,
            couplings,
            anti_map,
        };
    }

    /// Import a model in the [UFO 2.0](https://arxiv.org/abs/2304.09883) format. The specified `path` should point to
    /// the folder containing the Python source files.
    ///
    /// # Examples
    /// ```rust
    /// # use std::path::PathBuf;
    /// use feyngraph::Model;
    /// let model = Model::from_ufo(&PathBuf::from("tests/resources/Standard_Model_UFO")).unwrap();
    /// ```
    pub fn from_ufo(path: &Path) -> Result<Self, ModelError> {
        return ufo_parser::parse_ufo_model(path);
    }

    /// Import a model in [QGRAF's](http://cefema-gt.tecnico.ulisboa.pt/~paulo/qgraf.html) model format. The parser is
    /// not exhaustive in the options QGRAF supports and is only intended for backwards compatibility, especially for
    /// the models included in GoSam. UFO models should be preferred whenever possible.
    ///
    /// # Examples
    /// ```rust
    /// # use std::path::PathBuf;
    /// use feyngraph::Model;
    /// let model = Model::from_qgraf(&PathBuf::from("tests/resources/sm.qgraf")).unwrap();
    /// ```
    pub fn from_qgraf(path: &Path) -> Result<Self, ModelError> {
        return qgraf_parser::parse_qgraf_model(path);
    }

    /// Get the internal index of the anti-particle of the particle with the internal index `index`.
    pub fn get_anti_index(&self, index: usize) -> usize {
        return self.anti_map[index];
    }

    /// Get a reference to the anti-particle of the particle with the internal index `index`.
    pub fn get_anti(&self, index: usize) -> &Particle {
        return &self.particles[self.anti_map[index]];
    }

    /// Normalize the given internal index, i.e. return the given index if it belongs to a particle or return
    /// the index of the corresponding particle if the index of an anti-particle was given.
    pub fn normalize(&self, index: usize) -> usize {
        return if self.particles[index].pdg_code < 0 {
            self.get_anti_index(index)
        } else {
            index
        };
    }

    /// Get a reference to the particle with internal index `index`.
    pub fn get_particle(&self, index: usize) -> &Particle {
        return &self.particles[index];
    }

    /// Get a reference to the particle with name `name`.
    pub fn get_particle_by_name(&self, name: &str) -> Result<&Particle, ModelError> {
        return self
            .particles
            .get(name)
            .ok_or_else(|| ModelError::ContentError(format!("Particle '{}' not found in model", name)));
    }

    /// Get the internal index of the particle with name `name`
    pub fn get_particle_index(&self, name: &str) -> Result<usize, ModelError> {
        return self
            .particles
            .get_index_of(name)
            .ok_or_else(|| ModelError::ContentError(format!("Particle '{}' not found in model", name)));
    }

    /// Get a reference to the vertex with internal index `index`.
    pub fn vertex(&self, index: usize) -> &InteractionVertex {
        return &self.vertices[index];
    }

    /// Get an iterator over the interaction vertices.
    pub fn vertices_iter(&self) -> impl Iterator<Item = &InteractionVertex> {
        return self.vertices.values();
    }

    /// Get an iterator over the particles.
    pub fn particles_iter(&self) -> impl Iterator<Item = &Particle> {
        return self.particles.values();
    }

    /// Get the number of contained vertices.
    pub fn n_vertices(&self) -> usize {
        return self.vertices.len();
    }

    /// Get the names of the defined couplings.
    pub fn couplings(&self) -> &Vec<String> {
        return &self.couplings;
    }

    /// Check if adding `vertex` to the diagram is allowed by the maximum power of the coupling constants
    pub(crate) fn check_coupling_orders(
        &self,
        interaction: usize,
        remaining_coupling_orders: &Option<HashMap<String, usize>>,
    ) -> bool {
        return if let Some(remaining_orders) = remaining_coupling_orders {
            for (coupling, order) in self.vertices[interaction].coupling_orders() {
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

/// Reduced model object only containing topological properties.
///
/// This object can be constructed from a given physical [`Model`] or from a list of allowed node degrees.
///
/// # Examples
/// ```rust
/// use feyngraph::topology::TopologyModel;
/// let model = TopologyModel::from([3, 4, 5, 6]);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct TopologyModel {
    vertex_degrees: Vec<usize>,
}

impl TopologyModel {
    pub(crate) fn get(&self, i: usize) -> usize {
        return self.vertex_degrees[i];
    }

    /// Get an iterator over the allowed node degrees of the model.
    pub fn degrees_iter(&self) -> impl Iterator<Item = usize> {
        return self.vertex_degrees.clone().into_iter();
    }
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

impl<T> From<T> for TopologyModel
where
    T: Into<Vec<usize>>,
{
    fn from(degrees: T) -> Self {
        return Self {
            vertex_degrees: degrees.into(),
        };
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
