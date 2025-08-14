use crate::diagram::filter::DiagramSelector;
use crate::model::{Model, TopologyModel};
use crate::topology::{Topology, TopologyGenerator};
use itertools::Itertools;
use std::ops::Deref;
use std::sync::Arc;

use crate::diagram::view::DiagramView;
use crate::diagram::workspace::AssignWorkspace;
use crate::util::Error;
use crate::util::factorial;
use rayon::prelude::*;

mod components;
pub mod filter;
pub(crate) mod view;
mod workspace;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Leg {
    pub(crate) vertex: usize,
    pub(crate) particle: usize,
    pub(crate) momentum: Vec<i8>,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Propagator {
    pub(crate) vertices: [usize; 2],
    pub(crate) particle: usize,
    pub(crate) momentum: Vec<i8>,
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Vertex {
    pub(crate) propagators: Vec<isize>,
    pub(crate) interaction: usize,
}

impl Vertex {
    pub(crate) fn new(propagators: Vec<isize>, interaction: usize) -> Self {
        return Self {
            propagators,
            interaction,
        };
    }
}

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
                        .map(|edge| {
                            let prop = *edge as isize - workspace.topology.n_external as isize;
                            if prop >= 0
                                && propagators[prop as usize].vertices[0] == propagators[prop as usize].vertices[1]
                            {
                                vec![prop, prop]
                            } else {
                                vec![prop]
                            }
                        })
                        .flatten()
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

    pub fn count_opi_components(&self) -> usize {
        return self.bridges.len() + 1;
    }

    pub fn sign(&self) -> i8 {
        return self.sign;
    }

    pub fn symmetry_factor(&self) -> usize {
        return self.vertex_symmetry * self.propagator_symmetry;
    }

    pub fn n_in(&self) -> usize {
        return self.incoming_legs.len();
    }

    pub fn n_out(&self) -> usize {
        return self.outgoing_legs.len();
    }

    pub fn n_ext(&self) -> usize {
        return self.incoming_legs.len() + self.outgoing_legs.len();
    }
}

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

    pub fn len(&self) -> usize {
        return self.data.len();
    }

    pub fn is_empty(&self) -> bool {
        return self.data.is_empty();
    }

    pub fn get(&self, i: usize) -> DiagramView<'_> {
        return DiagramView::new(self.model.as_ref().unwrap(), &self.data[i], &self.momentum_labels);
    }

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
    pub fn new(
        incoming_particles: Vec<usize>,
        outgoing_particles: Vec<usize>,
        n_loops: usize,
        model: Model,
        selector: Option<DiagramSelector>,
    ) -> Self {
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
        return Self {
            model: Arc::new(model),
            selector: used_selector,
            incoming_particles,
            outgoing_particles: outgoing,
            n_external,
            n_loops,
            momentum_labels: None,
        };
    }

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

    pub fn assign_topology(&self, topology: &Topology) -> DiagramContainer {
        let mut assign_workspace = AssignWorkspace::new(
            topology,
            self.model.clone(),
            &self.selector,
            &self.incoming_particles,
            &self.outgoing_particles,
        );
        let mut container = assign_workspace.assign();
        container.model = Some(Arc::new(self.model.as_ref().clone()));
        return container;
    }

    pub fn assign_topologies(&self, topologies: &[Topology]) -> DiagramContainer {
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
        return container;
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
        selector.add_opi_count(1);
        let particles_in = vec![model.get_particle_index("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_index("G").unwrap().clone()];
        let generator = DiagramGenerator::new(particles_in, particle_out, 3, model, Some(selector));
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
        let particles_in = vec![model.get_particle_index("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_index("G").unwrap().clone()];
        let generator = DiagramGenerator::new(particles_in, particle_out, 3, model, Some(selector));
        let diagrams = generator.assign_topologies(&topologies);
        assert_eq!(diagrams.len(), 951);
    }

    #[test]
    pub fn diagram_generator_sign_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_UFO")).unwrap();
        let particles_in = vec![model.get_particle_index("u").unwrap().clone(); 2];
        let particle_out = vec![model.get_particle_index("u").unwrap().clone(); 2];
        let generator = DiagramGenerator::new(particles_in, particle_out, 0, model, None);
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
        let particles_in = vec![model.get_particle_index("G").unwrap().clone(); 2];
        let particle_out = vec![
            model.get_particle_index("u").unwrap().clone(),
            model.get_particle_index("u~").unwrap().clone(),
        ];
        let generator = DiagramGenerator::new(particles_in, particle_out, 1, model, Some(selector));
        let diagrams = generator.assign_topology(&topologies[33]);
        println!("{}", &topologies[33]);
        assert_eq!(diagrams.len(), 4);
    }
}
