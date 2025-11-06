//! A physical model used for diagram generation and drawing.

use crate::util::{HashMap, IndexMap};
use itertools::Itertools;
use log::warn;
use std::borrow::Borrow;
use std::hash::Hash;
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
    pub(crate) spin: isize,
    pub(crate) color: isize,
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

    /// Get $2s$, where $s$ is the particle's spin.
    pub fn spin(&self) -> isize {
        return self.spin;
    }

    /// Get the size of the particle's color representation.
    pub fn color(&self) -> isize {
        return self.color;
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
            spin: -self.spin,
            color: -self.color,
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
        spin: isize,
        color: isize,
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
            spin,
            color,
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

    pub fn order<Q>(&self, coupling: &Q) -> usize
    where
        Q: Hash + Eq,
        String: Borrow<Q>,
    {
        return *self.coupling_orders.get(coupling).unwrap_or(&0);
    }

    /// Get the degree of the vertex, i.e. the number of particles attached to it.
    pub fn degree(&self) -> usize {
        return self.particles.len();
    }

    /// Add a new coupling to the interaction vertex or overwrite an existing one.
    pub fn add_coupling(&mut self, coupling: impl Into<String> + Clone, power: usize) {
        match self.coupling_orders.insert(coupling.clone().into(), power) {
            None => (),
            Some(c) => warn!(
                "Vertex already has power {} in coupling {}, overwriting.",
                c,
                coupling.into()
            ),
        }
    }

    /// Check whether the given particle names match the interaction. "_" can be used as a wildcard to
    /// match all particles.
    pub fn match_particles<'q, S>(&self, query: impl Iterator<Item = &'q S>) -> bool
    where
        S: 'q + PartialEq<String> + Ord,
    {
        let particles_sorted: Vec<&String> = self.particles.iter().sorted().collect();
        let mut wildcards: usize = 0;
        let query_sorted: Vec<&S> = query
            .filter(|s| {
                if **s != "_".to_owned() {
                    true
                } else {
                    wildcards += 1;
                    false
                }
            })
            .sorted()
            .collect();
        if particles_sorted.len() != wildcards + query_sorted.len() {
            return false;
        }
        let mut query_cursor: usize = 0;
        for p in particles_sorted {
            if query_cursor < query_sorted.len() && *query_sorted[query_cursor] == *p {
                query_cursor += 1;
            } else {
                if wildcards > 0 {
                    wildcards -= 1;
                } else {
                    return false;
                }
            }
        }
        return true;
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
    splittings: HashMap<String, HashMap<String, Vec<(usize, usize)>>>,
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
        splittings: HashMap<String, HashMap<String, Vec<(usize, usize)>>>,
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
            splittings,
            anti_map,
        };
    }

    /// Create a new model without any particles, vertices or couplings.
    pub fn empty() -> Self {
        return Self {
            particles: IndexMap::default(),
            vertices: IndexMap::default(),
            couplings: Vec::new(),
            splittings: HashMap::default(),
            anti_map: Vec::new(),
        };
    }

    /// Add a new particle with the given properties to the model or overwrite an existing one. If `name == anti_name`,
    /// the particle is automatically marked as its own anti particle. Otherwise, the corresponding anti particle is
    /// automatically also added to the model.
    pub fn add_particle<S: Into<String> + PartialEq + Clone>(
        &mut self,
        name: S,
        anti_name: S,
        spin: isize,
        color: isize,
        pdg_code: isize,
        texname: S,
        antitexname: S,
        linestyle: LineStyle,
        statistic: Statistic,
    ) {
        let p = Particle {
            name: name.clone().into(),
            anti_name: anti_name.clone().into(),
            spin,
            color,
            pdg_code,
            texname: texname.into(),
            antitexname: antitexname.into(),
            self_anti: name == anti_name,
            linestyle,
            statistic,
        };
        if p.self_anti {
            match self.particles.insert(name.clone().into(), p) {
                None => (),
                Some(_) => warn!("Particle {} already present in model, replacing.", name.into()),
            }
            self.anti_map.push(self.anti_map.len());
        } else {
            match self.particles.insert(name.clone().into(), p.clone()) {
                None => (),
                Some(_) => warn!("Particle {} already present in model, replacing.", name.clone().into()),
            }
            self.particles.insert(anti_name.clone().into(), p.into_anti());
            self.anti_map.push(self.anti_map.len() + 1);
            self.anti_map.push(self.anti_map.len() - 1);
        }
    }

    /// Add a new vertex with the given properties to the model or overwrite an existing one. The `i`-th entry of the
    /// `spin_map` must be the leg `j` to which leg `i` is spin-connected.
    pub fn add_vertex<S: Into<String> + PartialEq + Clone>(
        &mut self,
        name: S,
        particles: Vec<S>,
        spin_map: Vec<isize>,
        coupling_orders: HashMap<S, usize>,
    ) -> Result<(), ModelError> {
        for coupling in coupling_orders.keys() {
            if !self.couplings.contains(&coupling.clone().into()) {
                self.couplings.push(coupling.clone().into());
            }
        }
        let mut v = InteractionVertex::new(
            name.clone().into(),
            particles.into_iter().map(|s| s.into()).collect(),
            spin_map,
            HashMap::from_iter(coupling_orders.into_iter().map(|(k, v)| (k.into(), v))),
        );
        for p in v.particles_iter() {
            if !self.particles.contains_key(p) {
                return Err(ModelError::ContentError(format!("particle {p} not found in model")));
            }
        }
        v.build_counts(&self.particles);
        match self.vertices.insert(name.clone().into(), v) {
            None => (),
            Some(_) => warn!("Vertex {} already present in model, replacing.", name.into()),
        }
        Ok(())
    }

    /// Deduplicate vertices in the model, i.e. merge all vertices with identical particles, spin connection and
    /// coupling powers. Returns a hash map containing the new vertex and all vertices which were merged into it.
    pub fn merge_vertices(&mut self) -> IndexMap<String, Vec<String>> {
        let mut mergings = IndexMap::default();
        let mut merged_vertices = IndexMap::default();
        let mut i = 1;
        for (_, vertices) in self
            .vertices
            .values()
            .into_group_map_by(|v| {
                (
                    v.particles.clone(),
                    v.coupling_orders.clone().into_iter().collect_vec(),
                    v.spin_map.clone(),
                )
            })
            .into_iter()
            .sorted_by_key(|(x, _)| x.clone())
        {
            if vertices.len() > 1 {
                mergings.insert(format!("V_M_{}", i), vertices.iter().map(|v| v.name.clone()).collect());
                merged_vertices.insert(
                    format!("V_M_{}", i),
                    InteractionVertex {
                        name: format!("V_M_{}", i),
                        particles: vertices[0].particles.clone(),
                        spin_map: vertices[0].spin_map.clone(),
                        coupling_orders: vertices[0].coupling_orders.clone(),
                        particle_counts: vertices[0].particle_counts.clone(),
                    },
                );
                i += 1;
            } else {
                merged_vertices.insert(vertices[0].name.clone(), vertices[0].clone());
            }
        }
        self.vertices = merged_vertices;
        return mergings;
    }

    /// Add a new coupling to the interaction vertex `vertex` or overwrite an existing one.
    pub fn add_coupling<S: Into<String> + Clone>(
        &mut self,
        vertex: S,
        coupling: S,
        power: usize,
    ) -> Result<(), ModelError> {
        match self.vertices.get_mut(&vertex.clone().into()) {
            Some(v) => v.add_coupling(coupling, power),
            None => {
                return Err(ModelError::ContentError(format!(
                    "vertex {} not found in model",
                    vertex.into()
                )));
            }
        }
        Ok(())
    }

    /// Split the existing vertex `vertex` into new vertices with names `new_vertices`.
    pub fn split_vertex<S: Into<String> + PartialEq + Clone>(
        &mut self,
        vertex: S,
        new_vertices: &[S],
    ) -> Result<(), ModelError> {
        let v = self.vertices.shift_remove(&vertex.clone().into());
        match v {
            None => {
                return Err(ModelError::ContentError(format!(
                    "vertex {} not found in model",
                    vertex.into()
                )));
            }
            Some(v) => {
                for name in new_vertices.iter() {
                    let mut new_vertex = v.clone();
                    new_vertex.name = name.clone().into();
                    match self.vertices.insert(name.clone().into(), new_vertex) {
                        None => (),
                        Some(_) => warn!("Vertex {} already present in model, replacing.", name.clone().into()),
                    }
                }
            }
        }
        Ok(())
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

    /// Get the splitting of the _original_ vertex `name`. Returns `None` if the requested vertex was not split up or
    /// does not exist in the model. If it was split, a hash map containing the vertices into which it was split is
    /// returned. For each vertex, the hash map contains a list of the `(color_index, lorentz_index)` tuples assigned
    /// to the created vertex.
    pub fn get_splitting(&self, name: &String) -> Option<&HashMap<String, Vec<(usize, usize)>>> {
        return self.splittings.get(name);
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
    use pretty_assertions::assert_eq;
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
