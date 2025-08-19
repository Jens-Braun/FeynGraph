//! Central module for the generation of Feynman diagrams.
//!
//! This module contains the [`DiagramGenerator`], which handles the generation of Feynman diagrams given a
//! [`Model`] and optionally a [`DiagramSelector`] restricting which diagrams are generated.

use crate::model::{Model, ModelError, TopologyModel};
use crate::topology::{Topology, TopologyGenerator};
pub use filter::DiagramSelector;
use itertools::Itertools;
use std::ops::Deref;
use std::sync::Arc;

use crate::diagram::view::DiagramView;
use crate::diagram::workspace::AssignWorkspace;
use crate::util::Error;
use crate::util::factorial;
use rayon::prelude::*;

mod components;
pub(crate) mod filter;
pub mod view;
mod workspace;

/// Internal representation of an external leg.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Leg {
    /// ID of the vertex the leg is attached to.
    pub vertex: usize,
    /// Internal ID of the particle assigned to the leg.
    pub particle: usize,
    /// Internal representation of the momentum assigned to the leg.
    pub momentum: Vec<i8>,
}

impl Leg {
    pub(crate) fn new(vertex: usize, particle: usize, momentum: Vec<i8>) -> Self {
        return Self {
            vertex,
            particle,
            momentum,
        };
    }
}

/// Internal representation of an internal propagator.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Propagator {
    /// Internal IDs of the vertices connected by the propagator.
    pub vertices: [usize; 2],
    /// Internal ID of the particle assigned to the propagator.
    pub particle: usize,
    /// Internal representation of the momentum assigned to the propagator.
    pub momentum: Vec<i8>,
}

impl Propagator {
    pub(crate) fn new(vertices: [usize; 2], particle: usize, momentum: Vec<i8>) -> Self {
        return Self {
            vertices,
            particle,
            momentum,
        };
    }
}

/// Internal representation of an internal vertex.
#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    /// Internal IDs of the propagators attached to the vertex.
    pub propagators: Vec<isize>,
    /// Internal ID of the interaction assigned to this vertex.
    pub interaction: usize,
}

impl Vertex {
    pub(crate) fn new(propagators: Vec<isize>, interaction: usize) -> Self {
        return Self {
            propagators,
            interaction,
        };
    }
}

/// Internal representation of a Feynman diagram.
///
/// This object is designed to be reasonably small in memory and therefore only contains the minimal information
/// needed to specify the diagram. The public interface to inspect a diagram lives at [`DiagramView`].
#[derive(Debug, PartialEq, Clone)]
pub struct Diagram {
    /// List of the diagram's incoming legs
    pub(crate) incoming_legs: Vec<Leg>,
    /// List of the diagram's outgoing legs
    pub(crate) outgoing_legs: Vec<Leg>,
    /// List of the diagram's propagators
    pub(crate) propagators: Vec<Propagator>,
    /// List of the diagram's vertices
    pub(crate) vertices: Vec<Vertex>,
    /// Symmetry factor due to vertex exchanges
    pub(crate) vertex_symmetry: usize,
    /// Symmetry factor due to propagator exchanges
    pub(crate) propagator_symmetry: usize,
    /// List of IDs of the bridge propagators
    pub(crate) bridges: Vec<usize>,
    /// Sign of the diagram
    pub(crate) sign: i8,
}

impl Diagram {
    fn from(workspace: &AssignWorkspace, vertex_symmetry: usize) -> Self {
        let incoming_legs = workspace
            .propagator_candidates
            .iter()
            .enumerate()
            .take(workspace.incoming_particles.len())
            .map(|(i, candidate)| {
                Leg::new(
                    workspace.topology.get_edge(i).connected_nodes[1] - workspace.topology.n_external,
                    candidate.particle.unwrap(),
                    workspace.topology.get_edge(i).momenta.as_ref().unwrap().clone(),
                )
            })
            .collect_vec();
        let outgoing_legs = workspace
            .propagator_candidates
            .iter()
            .enumerate()
            .skip(workspace.incoming_particles.len())
            .take(workspace.outgoing_particles.len())
            .map(|(i, candidate)| {
                Leg::new(
                    workspace.topology.get_edge(i).connected_nodes[1] - workspace.topology.n_external,
                    candidate.particle.unwrap(),
                    workspace.topology.get_edge(i).momenta.as_ref().unwrap().clone(),
                )
            })
            .collect_vec();
        let propagators = workspace
            .propagator_candidates
            .iter()
            .enumerate()
            .skip(workspace.topology.n_external)
            .map(|(i, candidate)| {
                Propagator::new(
                    [
                        workspace.topology.get_edge(i).connected_nodes[0] - workspace.topology.n_external,
                        workspace.topology.get_edge(i).connected_nodes[1] - workspace.topology.n_external,
                    ],
                    candidate.particle.unwrap(),
                    workspace
                        .topology
                        .get_edge(i)
                        .momenta
                        .as_ref()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, x)| {
                            if i >= workspace.incoming_particles.len() && i < workspace.topology.n_external {
                                -*x
                            } else {
                                *x
                            }
                        })
                        .collect_vec(),
                )
            })
            .collect_vec();
        let mut propagator_symmetry = 1;
        for ((vertices, _), count) in propagators.iter().counts_by(|prop| (prop.vertices, prop.particle)) {
            if vertices[0] == vertices[1] {
                propagator_symmetry *= 2_usize.pow(count as u32);
                propagator_symmetry *= factorial(count);
            } else {
                propagator_symmetry *= factorial(count);
            }
        }
        let vertices = workspace
            .vertex_candidates
            .iter()
            .skip(workspace.topology.n_external)
            .map(|candidate| {
                Vertex::new(
                    candidate
                        .edges
                        .iter()
                        .flat_map(|edge| {
                            let prop = *edge as isize - workspace.topology.n_external as isize;
                            if prop >= 0
                                && propagators[prop as usize].vertices[0] == propagators[prop as usize].vertices[1]
                            {
                                vec![prop, prop]
                            } else {
                                vec![prop]
                            }
                        })
                        .collect_vec(),
                    candidate.candidates[0],
                )
            })
            .collect_vec();
        let mut d = Diagram {
            incoming_legs,
            outgoing_legs,
            propagators,
            vertices,
            vertex_symmetry,
            propagator_symmetry,
            bridges: workspace
                .topology
                .bridges
                .iter()
                .map(|(v, w)| {
                    workspace
                        .topology
                        .edges
                        .iter()
                        .enumerate()
                        .find_map(|(i, edge)| {
                            if edge.connected_nodes == [*v, *w] || edge.connected_nodes == [*w, *v] {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .unwrap()
                })
                .collect_vec(),
            sign: 1,
        };
        d.sign = DiagramView::new(workspace.model.as_ref(), &d, workspace.momentum_labels.as_ref()).calculate_sign();
        return d;
    }

    /// Count the number of one-particle-irreducible components of the diagram.
    pub fn count_opi_components(&self) -> usize {
        return self.bridges.len() + 1;
    }

    /// Get the diagram's sign.
    pub fn sign(&self) -> i8 {
        return self.sign;
    }

    /// Get the diagram's symmetry factor.
    pub fn symmetry_factor(&self) -> usize {
        return self.vertex_symmetry * self.propagator_symmetry;
    }

    /// Get the number of incoming legs.
    pub fn n_in(&self) -> usize {
        return self.incoming_legs.len();
    }

    /// Ge the number of outgoing legs.
    pub fn n_out(&self) -> usize {
        return self.outgoing_legs.len();
    }

    /// Get the total number of external legs.
    pub fn n_ext(&self) -> usize {
        return self.incoming_legs.len() + self.outgoing_legs.len();
    }
}

/// Smart container for the Feynman diagrams generated by a [`DiagramGenerator`].
#[derive(Debug)]
pub struct DiagramContainer {
    pub(crate) model: Option<Arc<Model>>,
    pub(crate) momentum_labels: Arc<Vec<String>>,
    pub(crate) data: Vec<Diagram>,
}

impl DiagramContainer {
    pub(crate) fn new(model: Option<&Model>, momentum_labels: &[String]) -> Self {
        return if let Some(model) = model {
            Self {
                model: Some(Arc::new(model.clone())),
                momentum_labels: Arc::new(momentum_labels.to_owned()),
                data: Vec::new(),
            }
        } else {
            Self {
                model: None,
                momentum_labels: Arc::new(momentum_labels.to_owned()),
                data: Vec::new(),
            }
        };
    }
    pub(crate) fn with_capacity(model: Option<&Model>, momentum_labels: &[String], capacity: usize) -> Self {
        return if let Some(model) = model {
            Self {
                model: Some(Arc::new(model.clone())),
                momentum_labels: Arc::new(momentum_labels.to_owned()),
                data: Vec::with_capacity(capacity),
            }
        } else {
            Self {
                model: None,
                momentum_labels: Arc::new(momentum_labels.to_owned()),
                data: Vec::with_capacity(capacity),
            }
        };
    }

    fn inner_ref_mut(&mut self) -> &mut Vec<Diagram> {
        return &mut self.data;
    }

    /// Get the number of diagrams in the container.
    pub fn len(&self) -> usize {
        return self.data.len();
    }

    /// Check whether the container is empty.
    pub fn is_empty(&self) -> bool {
        return self.data.is_empty();
    }

    /// Get a [`DiagramView`] of the `i`-th diagram in the container.
    pub fn get(&self, i: usize) -> DiagramView<'_> {
        return DiagramView::new(self.model.as_ref().unwrap(), &self.data[i], &self.momentum_labels);
    }

    /// Get an iterator over the [`DiagramView`]'s in the container.
    pub fn views(&self) -> impl Iterator<Item = DiagramView<'_>> {
        return self
            .data
            .iter()
            .map(|d| DiagramView::new(self.model.as_ref().unwrap(), d, &self.momentum_labels));
    }

    /// Search for diagrams which would be selected by `selector`. Returns the index of the first selected diagram
    /// or `None` if no diagram is selected.
    pub fn query(&self, selector: &DiagramSelector) -> Option<usize> {
        return if let Some((i, _)) = self.data.iter().find_position(|diagram| {
            selector.select(
                self.model.as_ref().unwrap().clone(),
                self.momentum_labels.clone(),
                diagram,
            )
        }) {
            Some(i)
        } else {
            None
        };
    }
}

impl From<Vec<DiagramContainer>> for DiagramContainer {
    fn from(containers: Vec<DiagramContainer>) -> Self {
        if containers.is_empty() {
            return DiagramContainer::new(None, &[]);
        }
        let mut result = DiagramContainer::with_capacity(
            containers[0].model.as_deref(),
            &containers[0].momentum_labels,
            containers.iter().map(|x| x.data.len()).sum(),
        );
        for mut container in containers {
            result.inner_ref_mut().append(&mut container.data);
        }
        return result;
    }
}

impl Deref for DiagramContainer {
    type Target = Vec<Diagram>;
    fn deref(&self) -> &Self::Target {
        return &self.data;
    }
}

/// The central object for Feynman diagram generation, producing all possible diagrams for a given process.
///
/// # Examples
/// ```rust
/// use feyngraph::{Model, DiagramGenerator, DiagramSelector};
///
/// let model = Model::default();
/// let mut selector = DiagramSelector::new();
/// selector.select_on_shell();
/// selector.select_self_loops(0);
/// let generator = DiagramGenerator::new(&["g", "g"], &["g", "g", "g"], 2, model, Some(selector)).unwrap();
/// let diags = generator.generate();
/// assert_eq!(diags.len(), 183350);
/// ```
pub struct DiagramGenerator {
    model: Arc<Model>,
    selector: DiagramSelector,
    incoming_particles: Vec<usize>,
    outgoing_particles: Vec<usize>,
    n_external: usize,
    n_loops: usize,
    momentum_labels: Option<Vec<String>>,
}

impl DiagramGenerator {
    /// Create a new diagram generator from a process specification:
    ///
    /// - `particles_in`: list of names of incoming particles
    /// - `particles_out`: list of names of outgoing particles
    /// - `n_loops`: number of loops per diagram
    /// - `model`: physical [`Model`]
    /// - `selector`\[optional\]: [`DiagramSelector`] restricting the generated diagrams
    pub fn new(
        particles_in: &[&str],
        particles_out: &[&str],
        n_loops: usize,
        model: Model,
        selector: Option<DiagramSelector>,
    ) -> Result<Self, ModelError> {
        let incoming_particles: Vec<usize> = particles_in
            .iter()
            .map(|p| model.get_particle_index(p))
            .collect::<Result<_, _>>()?;
        let outgoing_particles: Vec<usize> = particles_out
            .iter()
            .map(|p| model.get_particle_index(p))
            .collect::<Result<_, _>>()?;
        let n_external = incoming_particles.len() + outgoing_particles.len();
        let outgoing = outgoing_particles
            .into_iter()
            .map(|p| model.get_anti_index(p))
            .collect_vec();
        let used_selector;
        if let Some(selector) = selector {
            used_selector = selector;
        } else {
            used_selector = DiagramSelector::default();
        }
        return Ok(Self {
            model: Arc::new(model),
            selector: used_selector,
            incoming_particles,
            outgoing_particles: outgoing,
            n_external,
            n_loops,
            momentum_labels: None,
        });
    }

    /// Change the momentum labels used when rendering momenta as strings.
    pub fn set_momentum_labels(&mut self, labels: Vec<String>) -> Result<(), Error> {
        if !labels.len() == self.n_external + self.n_loops {
            return Err(Error::InputError(format!(
                "Found {} momenta, but n_external + n_loops = {} are required",
                labels.len(),
                self.n_external + self.n_loops
            )));
        }
        self.momentum_labels = Some(labels);
        return Ok(());
    }

    /// Generate the Feynman diagrams for the given setup.
    pub fn generate(&self) -> DiagramContainer {
        let mut topo_generator = TopologyGenerator::new(
            self.n_external,
            self.n_loops,
            TopologyModel::from(self.model.as_ref()),
            Some(self.selector.as_topology_selector()),
        );
        if let Some(ref labels) = self.momentum_labels {
            topo_generator.set_momentum_labels(labels.clone()).unwrap();
        }
        let topologies = topo_generator.generate();
        let mut containers: Vec<DiagramContainer> = Vec::new();
        topologies
            .inner_ref()
            .into_par_iter()
            .map(|topology| {
                let mut assign_workspace = AssignWorkspace::new(
                    topology,
                    self.model.clone(),
                    &self.selector,
                    &self.incoming_particles,
                    &self.outgoing_particles,
                );
                return assign_workspace.assign();
            })
            .collect_into_vec(&mut containers);
        let mut container = DiagramContainer::from(containers);
        container.model = Some(Arc::new(self.model.as_ref().clone()));
        return container;
    }

    /// Generate the Feynman diagrams for the given setup without saving them, only returning the total number of
    /// diagrams.
    pub fn count(&self) -> usize {
        let mut topo_generator = TopologyGenerator::new(
            self.n_external,
            self.n_loops,
            TopologyModel::from(self.model.as_ref()),
            Some(self.selector.as_topology_selector()),
        );
        if let Some(ref labels) = self.momentum_labels {
            topo_generator.set_momentum_labels(labels.clone()).unwrap();
        }
        let topologies = topo_generator.generate();
        let mut counts: Vec<usize> = Vec::new();
        topologies
            .inner_ref()
            .into_par_iter()
            .map(|topology| {
                let mut assign_workspace = AssignWorkspace::new(
                    topology,
                    self.model.clone(),
                    &self.selector,
                    &self.incoming_particles,
                    &self.outgoing_particles,
                );
                return assign_workspace.assign().len();
            })
            .collect_into_vec(&mut counts);
        return counts.into_iter().sum();
    }

    /// Produce Feynman diagrams by assigning particles to a given topology `topology`.
    pub fn assign_topology(&self, topology: &Topology) -> Result<DiagramContainer, Error> {
        if self.n_external != topology.n_external {
            return Err(Error::InputError(format!(
                "expected topologies with {} legs, found topology with {} legs",
                self.n_external, topology.n_external
            )));
        }
        if self.n_loops != topology.n_loops {
            return Err(Error::InputError(format!(
                "expected topologies with {} loops, found topology with {} loops",
                self.n_loops, topology.n_loops
            )));
        }
        let mut assign_workspace = AssignWorkspace::new(
            topology,
            self.model.clone(),
            &self.selector,
            &self.incoming_particles,
            &self.outgoing_particles,
        );
        let mut container = assign_workspace.assign();
        container.model = Some(Arc::new(self.model.as_ref().clone()));
        return Ok(container);
    }

    /// Produce Feynman diagrams by assigning particles to a given set of topologies `topologies`.
    pub fn assign_topologies(&self, topologies: &[Topology]) -> Result<DiagramContainer, Error> {
        for topology in topologies {
            if self.n_external != topology.n_external {
                return Err(Error::InputError(format!(
                    "expected topologies with {} legs, found topology with {} legs",
                    self.n_external, topology.n_external
                )));
            }
            if self.n_loops != topology.n_loops {
                return Err(Error::InputError(format!(
                    "expected topologies with {} loops, found topology with {} loops",
                    self.n_loops, topology.n_loops
                )));
            }
        }
        let mut containers: Vec<DiagramContainer> = Vec::new();
        topologies
            .into_par_iter()
            .map(|topology| {
                let mut assign_workspace = AssignWorkspace::new(
                    topology,
                    self.model.clone(),
                    &self.selector,
                    &self.incoming_particles,
                    &self.outgoing_particles,
                );
                return assign_workspace.assign();
            })
            .collect_into_vec(&mut containers);
        let mut container = DiagramContainer::from(containers);
        container.model = Some(Arc::new(self.model.as_ref().clone()));
        return Ok(container);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::filter::TopologySelector;
    use std::path::PathBuf;
    use std::sync::Arc;
    use test_log::test;

    #[test]
    pub fn diagram_generator_qcd_g_prop_opi_3loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_UFO")).unwrap();
        let mut selector = DiagramSelector::default();
        selector.select_opi_components(1);
        let particles_in = ["G"];
        let particle_out = ["G"];
        let generator = DiagramGenerator::new(&particles_in, &particle_out, 3, model, Some(selector)).unwrap();
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 479);
    }

    #[test]
    pub fn diagram_generator_qcd_g_prop_no_self_loops_3loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_UFO")).unwrap();
        let mut topo_selector = TopologySelector::new();
        topo_selector.add_custom_function(Arc::new(|topo: &Topology| -> bool {
            !topo
                .edges_iter()
                .any(|edge| edge.connected_nodes[0] == edge.connected_nodes[1])
        }));
        let topo_generator = TopologyGenerator::new(2, 3, (&model).into(), Some(topo_selector));
        let topologies = topo_generator.generate();
        let selector = DiagramSelector::default();
        let particles_in = ["G"];
        let particle_out = ["G"];
        let generator = DiagramGenerator::new(&particles_in, &particle_out, 3, model, Some(selector)).unwrap();
        let diagrams = generator.assign_topologies(&topologies).unwrap();
        assert_eq!(diagrams.len(), 951);
    }

    #[test]
    pub fn diagram_generator_sign_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_UFO")).unwrap();
        let particles_in = ["u", "u"];
        let particle_out = ["u", "u"];
        let generator = DiagramGenerator::new(&particles_in, &particle_out, 0, model, None).unwrap();
        let diags = generator.generate();
        assert_eq!(diags.len(), 2);
        assert_eq!(diags[0].sign, -diags[1].sign);
    }

    #[test]
    fn diagram_generator_sign_1l_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_UFO")).unwrap();
        let mut topo_selector = TopologySelector::new();
        topo_selector.add_custom_function(Arc::new(|topo: &Topology| -> bool {
            !topo
                .edges_iter()
                .any(|edge| edge.connected_nodes[0] == edge.connected_nodes[1])
        }));
        let topo_generator = TopologyGenerator::new(4, 1, (&model).into(), Some(topo_selector));
        let topologies = topo_generator.generate();
        let selector = DiagramSelector::default();
        let particles_in = ["G", "G"];
        let particle_out = ["u", "u~"];
        let generator = DiagramGenerator::new(&particles_in, &particle_out, 1, model, Some(selector)).unwrap();
        let diagrams = generator.assign_topology(&topologies[33]).unwrap();
        println!("{}", &topologies[33]);
        assert_eq!(diagrams.len(), 4);
    }
}
