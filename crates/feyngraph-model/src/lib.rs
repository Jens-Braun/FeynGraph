use itertools::Itertools;
use std::{
    error::Error,
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    path::Path,
};

use util::{HashMap, IndexMap};

mod particle;
mod qgraf;
mod ufo;
mod vertex;

pub use particle::{LineStyle, ParticleBase, ParticleColor, ParticleDraw};
pub use ufo::{UFOModel, UFOParticle, UFOVertex};
pub use vertex::VertexBase;

use crate::qgraf::QGRAFModel;

#[derive(Debug)]
pub struct ModelError {
    msg: String,
    kind: ErrorKind,
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            ErrorKind::ParticleNotFound | ErrorKind::VertexNotFound => None,
            ErrorKind::IO(err) => Some(err),
            ErrorKind::ParseError(err) => Some(err.as_ref()),
        }
    }
}

impl ModelError {
    pub fn particle_not_found(p: impl Display) -> Self {
        Self {
            msg: format!("Particle `{}` not found in model", p),
            kind: ErrorKind::ParticleNotFound,
        }
    }

    pub fn vertex_not_found(p: impl Display) -> Self {
        Self {
            msg: format!("Vertex `{}` not found in model", p),
            kind: ErrorKind::VertexNotFound,
        }
    }

    pub fn io(file: impl Display, source: std::io::Error) -> Self {
        Self {
            msg: format!("IO error while reading file `{file}`"),
            kind: ErrorKind::IO(source),
        }
    }

    pub fn parse_error(file: impl Display, source: Box<dyn Error>) -> Self {
        Self {
            msg: format!("Error while parsing file `{file}`"),
            kind: ErrorKind::ParseError(source),
        }
    }
}

impl Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

#[derive(Debug)]
pub enum ErrorKind {
    ParticleNotFound,
    VertexNotFound,
    IO(std::io::Error),
    ParseError(Box<dyn Error>),
}

pub trait ModelBase: Debug + Send + Sync {
    type Particle: ParticleBase;
    type Vertex: VertexBase;

    fn particle(&self, index: usize) -> &Self::Particle;
    fn particle_by_name(&self, name: impl AsRef<str>) -> Result<&Self::Particle, ModelError>;
    fn particle_index_by_name(&self, name: impl AsRef<str>) -> Result<usize, ModelError>;
    fn anti_particle(&self, particle: usize) -> &Self::Particle;
    fn anti_particle_index(&self, particle: usize) -> usize;

    fn vertex(&self, index: usize) -> &Self::Vertex;

    fn particles(&self) -> impl ExactSizeIterator<Item = &Self::Particle>;
    fn vertices(&self) -> impl ExactSizeIterator<Item = &Self::Vertex>;
    fn couplings(&self) -> impl ExactSizeIterator<Item = impl AsRef<str>>;

    fn supports_sign(&self) -> bool;

    /// Check if adding `vertex` to the diagram is allowed by the maximum power of the coupling constants
    fn check_coupling_orders(
        &self,
        interaction: usize,
        remaining_coupling_orders: &Option<HashMap<String, usize>>,
    ) -> bool {
        return if let Some(remaining_orders) = remaining_coupling_orders {
            for (coupling, order) in self.vertex(interaction).coupling_orders() {
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

#[derive(Debug, Clone)]
pub struct Model<M> {
    base: M,
    particle_counts: IndexMap<usize, Vec<(usize, usize)>>,
}

impl Model<QGRAFModel> {
    pub fn qgraf(path: &Path) -> Result<Self, ModelError> {
        Ok(Self::from_base(QGRAFModel::parse(path)?))
    }
}

impl Model<UFOModel> {
    pub fn ufo(path: &Path) -> Result<Self, ModelError> {
        Ok(Self::from_base(UFOModel::parse(path)?))
    }

    pub fn sm() -> Self {
        Self::from_base(UFOModel::sm())
    }

    pub fn add_vertex<S: Into<String> + AsRef<str> + PartialEq + Clone>(
        &mut self,
        name: S,
        particles: Vec<S>,
        spin_map: Vec<isize>,
        coupling_orders: HashMap<S, usize>,
    ) -> Result<(), ModelError> {
        self.base.add_vertex(name, particles, spin_map, coupling_orders)?;
        self.rebuild_cache();
        Ok(())
    }

    pub fn merge_vertices(&mut self) -> IndexMap<String, Vec<String>> {
        let res = self.base.merge_vertices();
        self.rebuild_cache();
        res
    }
}

impl<M: ModelBase> Model<M> {
    pub fn from_base(base: M) -> Self {
        let counts = base
            .vertices()
            .enumerate()
            .map(|(v_i, v)| {
                (
                    v_i,
                    v.particles()
                        .iter()
                        .counts_by(|p| p.as_ref())
                        .into_iter()
                        .map(|(k, v)| (base.particle_index_by_name(k).unwrap(), v))
                        .collect(),
                )
            })
            .collect();
        Self {
            base,
            particle_counts: counts,
        }
    }

    pub fn rebuild_cache(&mut self) {
        self.particle_counts = self
            .base
            .vertices()
            .enumerate()
            .map(|(v_i, v)| {
                (
                    v_i,
                    v.particles()
                        .iter()
                        .counts_by(|p| p.as_ref())
                        .into_iter()
                        .map(|(k, v)| (self.base.particle_index_by_name(k).unwrap(), v))
                        .collect(),
                )
            })
            .collect();
    }

    pub fn particle_count(&self, vertex: usize, particle: usize) -> usize {
        match self.particle_counts.get(&vertex) {
            None => 0,
            Some(v) => v
                .iter()
                .find_map(|(p, c)| if *p == particle { Some(*c) } else { None })
                .unwrap_or(0),
        }
    }

    pub fn particle_counts(&self, vertex: usize) -> &[(usize, usize)] {
        match self.particle_counts.get(&vertex) {
            None => &[],
            Some(v) => v,
        }
    }
}

impl<M: ModelBase> Deref for Model<M> {
    type Target = M;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<M: ModelBase> DerefMut for Model<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
/// Reduced model object only containing topological properties.
///
/// This object can be constructed from a given physical [`Model`] or from a list of allowed node degrees.
///
/// # Examples
/// ```rust
/// # use feyngraph_model::TopologyModel;
/// let model = TopologyModel::from([3, 4, 5, 6]);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct TopologyModel {
    vertex_degrees: Vec<usize>,
}

impl TopologyModel {
    pub fn get(&self, i: usize) -> usize {
        return self.vertex_degrees[i];
    }

    /// Get an iterator over the allowed node degrees of the model.
    pub fn degrees_iter(&self) -> impl Iterator<Item = usize> {
        return self.vertex_degrees.clone().into_iter();
    }
}

impl<M: ModelBase> From<&Model<M>> for TopologyModel {
    fn from(model: &Model<M>) -> Self {
        let mut vertex_degrees = Vec::new();
        for vertex in model.vertices() {
            vertex_degrees.push(vertex.particles().len());
        }
        return Self {
            vertex_degrees: vertex_degrees.into_iter().sorted().dedup().collect_vec(),
        };
    }
}

impl<M: ModelBase> From<Model<M>> for TopologyModel {
    fn from(model: Model<M>) -> Self {
        let mut vertex_degrees = Vec::new();
        for vertex in model.vertices() {
            vertex_degrees.push(vertex.particles().len());
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
    use super::{Model, TopologyModel};
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;
    use test_log::test;

    #[test]
    fn model_conversion_test() {
        let model = Model::ufo(&PathBuf::from("../../tests/models/Standard_Model_UFO")).unwrap();
        let topology_model = TopologyModel::from(&model);
        assert_eq!(
            topology_model,
            TopologyModel {
                vertex_degrees: vec![3, 4]
            }
        );
    }
}
