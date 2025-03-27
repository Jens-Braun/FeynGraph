use std::ops::Deref;
use std::fmt::Write;
use itertools::Itertools;
use crate::diagram::filter::DiagramSelector;
use crate::model::{Model, Particle, TopologyModel, InteractionVertex};
use crate::topology::{Topology, TopologyGenerator};

use rayon::prelude::*;
use crate::diagram::workspace::AssignWorkspace;
use crate::util::Error;
use crate::util::factorial;

mod components;
pub mod filter;
mod workspace;
mod view;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Propagator {
    pub(crate) vertices: (usize, usize),
    pub(crate) particle: Particle,
    pub(crate) momentum: Vec<i8>
}

impl Propagator {
    pub fn new(vertices: (usize, usize), particle: Particle, momentum: Vec<i8>) -> Self {
        return Self {
            vertices,
            particle,
            momentum
        }
    }

    pub(crate) fn invert(self) -> Self {
        return Self {
            vertices: (self.vertices.1, self.vertices.0),
            particle: self.particle.into_anti(),
            momentum: self.momentum.into_iter().map(|x| -x).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    pub(crate) propagators: Vec<usize>,
    pub(crate) interaction: InteractionVertex,
}

impl Vertex {
    pub fn new(propagators: Vec<usize>, interaction: InteractionVertex) -> Self {
        return Self {
            propagators,
            interaction
        }
    }
}

impl std::fmt::Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[ ", self.interaction.name)?;
        for p in self.interaction.particles.iter() {
            write!(f, "{} ", p)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Diagram {
    incoming_particles: Vec<Particle>,
    outgoing_particles:Vec<Particle>,
    vertices: Vec<Vertex>,
    propagators: Vec<Propagator>,
    vertex_symmetry: usize,
    propagator_symmetry: usize,
    momentum_labels: Vec<String>,
    bridges: Vec<usize>,
    sign: i8
}

impl Diagram {
    fn from(workspace: &AssignWorkspace, vertex_symmetry: usize) -> Self {
        let propagators = workspace.propagator_candidates.iter().enumerate()
            .map(|(i, candidate)| {
                Propagator::new(
                    workspace.topology.get_edge(i).connected_nodes,
                    workspace.model.get_particle(candidate.particle.unwrap()).clone(),
                    workspace.topology.get_edge(i).momenta.as_ref().unwrap().iter().enumerate().map(
                        |(i, x)| if i >= workspace.incoming_particles.len()
                            && i < workspace.incoming_particles.len() + workspace.outgoing_particles.len() {
                            -*x
                        } else {
                            *x
                        }
                    ).collect_vec(),
                )
            }
            ).collect_vec();
        let mut propagator_symmetry = 1;
        for ((vertices, _), count) in
            propagators.iter().counts_by(|prop| (prop.vertices, prop.particle.get_pdg())) {
            if vertices.0 == vertices.1 {
                propagator_symmetry *= 2_usize.pow(count as u32);
                propagator_symmetry *= factorial(count);
            } else {
                propagator_symmetry *= factorial(count);
            }
        }
        return Diagram {
            incoming_particles: workspace.incoming_particles.clone(),
            outgoing_particles: workspace.outgoing_particles.clone(),
            vertices: workspace.vertex_candidates.iter()
                .filter_map(|candidate| {
                    if candidate.degree == 1 {
                        None
                    } else {
                        Some(Vertex::new(candidate.edges.clone(),
                                         workspace.model.get_vertex(candidate.candidates[0]).clone()
                        ))
                    }
                }).collect_vec(),
            propagators,
            vertex_symmetry,
            propagator_symmetry,
            momentum_labels: workspace.topology.momentum_labels.clone(),
            bridges: workspace.topology.bridges.iter().map(
                |(v, w)| {
                    workspace.topology.edges.iter().enumerate()
                        .find_map(|(i, edge)|
                            if edge.connected_nodes == (*v, *w)
                                || edge.connected_nodes == (*w, *v) {Some(i)} else {None}
                        ).unwrap()
                }).collect_vec(),
            sign: workspace.calculate_sign()
        };
    }

    pub fn momentum_string(&self, propagator_index: usize, invert: bool) -> String {
        let mut result = String::with_capacity(5*self.momentum_labels.len());
        let mut first: bool = true;
        let sign = if invert { -1 } else { 1 };
        for (i, coefficient) in self.propagators[propagator_index].momentum.iter().enumerate() {
            if *coefficient == 0 { continue; }
            if first {
                match *coefficient * sign {
                    1 => {
                        write!(&mut result, "{}", self.momentum_labels[i]).unwrap();
                    },
                    -1 => {
                        write!(&mut result, "-{}", self.momentum_labels[i]).unwrap();
                    },
                    x if x < 0 => write!(&mut result, "- {}*{}", x.abs(), self.momentum_labels[i]).unwrap(),
                    x => write!(&mut result, "{}*{}", x, self.momentum_labels[i]).unwrap()
                }
                first = false;
            } else {
                write!(&mut result, " ").unwrap();
                match *coefficient * sign {
                    1 => {
                        write!(&mut result, "+ {}", self.momentum_labels[i]).unwrap();
                    },
                    -1 => {
                        write!(&mut result, "- {}", self.momentum_labels[i]).unwrap();
                    },
                    x if x < 0 => write!(&mut result, "- {}*{}", x.abs(), self.momentum_labels[i]).unwrap(),
                    x => write!(&mut result, "+ {}*{}", x, self.momentum_labels[i]).unwrap()
                }
            }
        }
        return result;
    }

    pub fn propagators_iter(&self) -> impl Iterator<Item = &Propagator> {
        return self.propagators.iter();
    }

    pub fn incoming_iter(&self) -> impl Iterator<Item = &Propagator> {
        return self.propagators.iter().take(self.incoming_particles.len());
    }

    pub fn outgoing_iter(&self) -> impl Iterator<Item = &Propagator> {
        return self.propagators.iter().skip(self.incoming_particles.len()).take(self.outgoing_particles.len());
    }

    pub fn count_opi_components(&self) -> usize {
        return self.bridges.len() + 1;
    }

    pub fn get_sign(&self) -> i8 { return self.sign; }

    pub fn get_symmetry_factor(&self) -> usize { return self.vertex_symmetry * self.propagator_symmetry; }

    pub fn get_n_in(&self) -> usize { return self.incoming_particles.len(); }

    pub fn get_n_out(&self) -> usize { return self.outgoing_particles.len(); }

    pub fn get_n_ext(&self) -> usize { return self.incoming_particles.len() + self.outgoing_particles.len(); }

    pub fn vertices_iter(&self) -> impl Iterator<Item = &Vertex> { return self.vertices.iter(); }

    pub fn get_vertex(&self, vertex: usize) -> &Vertex { return &self.vertices[vertex-self.get_n_ext()]; }

    pub fn get_propagator(&self, propagator: usize) -> &Propagator { return &self.propagators[propagator]; }
}

impl std::fmt::Display for Diagram {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Diagram {{")?;
        write!(f, "    Process: ")?;
        for incoming in self.incoming_particles.iter() {
            write!(f, "{} ", incoming.get_name())?;
        }
        write!(f, "-> ")?;
        for outgoing in self.outgoing_particles.iter() {
            write!(f, "{} ", outgoing.get_name())?;
        }
        writeln!(f, "")?;
        write!(f, "    Vertices: [ ")?;
        for vertex in self.vertices.iter() {
            write!(f, "{} ", vertex)?;
        }
        writeln!(f, "]")?;
        writeln!(f, "    Propagators: [")?;
        for (i, propagator) in self.propagators.iter().enumerate() {
            writeln!(f, "        {}[{} -> {}], p = {},",
                     propagator.particle.get_name(),
                     propagator.vertices.0,
                     propagator.vertices.1,
                     self.momentum_string(i, false)
            )?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "    SymmetryFactor: 1/{}", self.vertex_symmetry * self.propagator_symmetry)?;
        writeln!(f, "    Sign: {}", if self.sign == 1 {"+"} else {"-"})?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct DiagramContainer {
    data: Vec<Diagram>,
}

impl DiagramContainer {
    
    pub fn new() -> Self {
        return Self { data: Vec::new() };
    }
    pub fn with_capacity(capacity: usize) -> Self {
        return Self {
            data: Vec::with_capacity(capacity),
        }
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

    pub fn get(&self, i: usize) -> &Diagram {
        return &self.data[i];
    }

    /// Search for topologies which would be selected by `selector`. Returns the index of the first selected diagram
    /// or `None` if no diagram is selected.
    pub fn query(&self, selector: &DiagramSelector) -> Option<usize> {
        return if let Some((i, _)) = self.data.iter().find_position(|diagram| selector.select(diagram)) {
            Some(i)
        } else {
            None
        }
    }
}

impl Default for DiagramContainer {
    fn default() -> Self {
        return Self { data: Vec::new() };
    }
}

impl From<Vec<DiagramContainer>> for DiagramContainer {
    fn from(containers: Vec<DiagramContainer>) -> Self {
        let mut result = DiagramContainer::with_capacity(
            containers.iter().map(|x| x.data.len()).sum()
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
    model: Model,
    selector: DiagramSelector,
    incoming_particles: Vec<Particle>,
    outgoing_particles: Vec<Particle>,
    n_external: usize,
    n_loops: usize,
    momentum_labels: Option<Vec<String>>,
}

impl DiagramGenerator {
    pub fn new(incoming_particles: Vec<Particle>, 
               outgoing_particles: Vec<Particle>, 
               n_loops: usize,
               model: Model,
               selector: Option<DiagramSelector>
    ) -> Self {
        let n_external = incoming_particles.len() + outgoing_particles.len();
        let used_selector;
        if let Some(selector) = selector {
            used_selector = selector;
        } else {
            used_selector = DiagramSelector::default();
        }
        return Self {
            model,
            selector: used_selector,
            incoming_particles,
            outgoing_particles,
            n_external,
            n_loops,
            momentum_labels: None,
        }
    }

    pub fn set_momentum_labels(&mut self, labels: Vec<String>) -> Result<(), Error> {
        if !labels.len() == self.n_external + self.n_loops {
            return Err(Error::InputError(
                format!("Found {} momenta, but n_external + n_loops = {} are required",
                        labels.len(), self.n_external + self.n_loops)
            ));
        }
        self.momentum_labels = Some(labels);
        return Ok(());
    }

    pub fn generate(&self) -> DiagramContainer {
        let mut topo_generator = TopologyGenerator::new(
            self.n_external,
            self.n_loops,
            TopologyModel::from(&self.model),
            Some(self.selector.as_topology_selector()),
        );
        if let Some(ref labels) = self.momentum_labels {
            topo_generator.set_momentum_labels(labels.clone()).unwrap();
        }
        let topologies = topo_generator.generate();
        let mut containers: Vec<DiagramContainer> = Vec::new();
        topologies.inner_ref().into_par_iter().map(|topology| {
            let mut assign_workspace = AssignWorkspace::new(
                topology,
                &self.model,
                &self.selector,
                &self.incoming_particles,
                &self.outgoing_particles
            );
            return assign_workspace.assign();
        }).collect_into_vec(&mut containers);
        return DiagramContainer::from(containers);
    }

    pub fn assign_topology(&self, topology: &Topology) -> DiagramContainer {
        let mut assign_workspace = AssignWorkspace::new(
            topology,
            &self.model,
            &self.selector,
            &self.incoming_particles,
            &self.outgoing_particles
        );
        return assign_workspace.assign();
    }

    pub fn assign_topologies(&self, topologies: &[Topology]) -> DiagramContainer {
        let mut containers: Vec<DiagramContainer> = Vec::new();
        topologies.into_par_iter().map(|topology| {
            let mut assign_workspace = AssignWorkspace::new(
                topology,
                &self.model,
                &self.selector,
                &self.incoming_particles,
                &self.outgoing_particles
            );
            return assign_workspace.assign();
        }).collect_into_vec(&mut containers);
        return DiagramContainer::from(containers);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use crate::topology::filter::TopologySelector;
    use super::*;
    
    #[test]
    pub fn diagram_generator_qcd_2g_2g_tree() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone(); 2];
        let particle_out = particles_in.clone();
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            0,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 4);
    }

    #[test]
    pub fn diagram_generator_qcd_2g_2u_tree() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone(); 2];
        let particle_out = vec![
            model.get_particle_name("u").unwrap().clone(),
            model.get_particle_name("u~").unwrap().clone()
        ];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            0,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 3);
    }

    #[test]
    pub fn diagram_generator_qcd_2g_2u_1loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone(); 2];
        let particle_out = vec![
            model.get_particle_name("u").unwrap().clone(),
            model.get_particle_name("u~").unwrap().clone()
        ];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            1,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 134);
    }

    #[test]
    pub fn diagram_generator_qcd_2g_2u_2loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone(); 2];
        let particle_out = vec![
            model.get_particle_name("u").unwrap().clone(),
            model.get_particle_name("u~").unwrap().clone()
        ];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            2,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 5569);
    }

    #[test]
    pub fn diagram_generator_qcd_2g_2u_3loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone(); 2];
        let particle_out = vec![
            model.get_particle_name("u").unwrap().clone(),
            model.get_particle_name("u~").unwrap().clone()
        ];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            3,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 233199);
    }

    #[test]
    pub fn diagram_generator_qcd_u_prop_2loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("u").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("u").unwrap().clone()];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            2,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 80);
    }

    #[test]
    pub fn diagram_generator_qcd_g_prop_2loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("G").unwrap().clone()];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            2,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 200);
    }

    #[test]
    pub fn diagram_generator_qcd_g_prop_3loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("G").unwrap().clone()];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            3,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 5386);
    }

    #[test]
    pub fn diagram_generator_qcd_g_prop_opi_3loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let mut selector = DiagramSelector::default();
        selector.add_opi_count(1);
        let particles_in = vec![model.get_particle_name("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("G").unwrap().clone()];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            3,
            model,
            Some(selector)
        );
        let diagrams = generator.generate();
        assert_eq!(diagrams.len(), 479);
    }

    #[test]
    pub fn diagram_generator_qcd_g_prop_no_self_loops_3loop() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let mut topo_selector = TopologySelector::new();
        topo_selector.add_custom_function(
            Arc::new(|topo: &Topology| -> bool {
                !topo.edges_iter().any(|edge| edge.connected_nodes.0 == edge.connected_nodes.1)
            })
        );
        let topo_generator = TopologyGenerator::new(2, 3, (&model).into(), Some(topo_selector));
        let topologies = topo_generator.generate();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("G").unwrap().clone()];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            3,
            model,
            Some(selector)
        );
        let diagrams = generator.assign_topologies(&topologies);
        assert_eq!(diagrams.len(), 951);
    }

    #[test]
    fn diagram_generator_sm_u_prop_1l_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/Standard_Model_UFO")).unwrap();
        let topo_selector = TopologySelector::new();
        let topo_generator = TopologyGenerator::new(2, 1, (&model).into(), Some(topo_selector));
        let topologies = topo_generator.generate();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("u").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("u").unwrap().clone()];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            1,
            model,
            Some(selector)
        );
        let diagrams = generator.assign_topology(&topologies[1]);
        assert_eq!(diagrams.len(), 37);
    }

    #[test]
    pub fn diagram_generator_sign_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let particles_in = vec![model.get_particle_name("u").unwrap().clone(); 2];
        let particle_out = vec![model.get_particle_name("u").unwrap().clone(); 2];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            0,
            model,
            None
        );
        let diags = generator.generate();
        assert_eq!(diags.len(), 2);
        assert_eq!(diags[0].sign, -diags[1].sign);
    }

    #[test]
    fn diagram_generator_sign_1l_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let mut topo_selector = TopologySelector::new();
        topo_selector.add_custom_function(
            Arc::new(|topo: &Topology| -> bool {
                !topo.edges_iter().any(|edge| edge.connected_nodes.0 == edge.connected_nodes.1)
            })
        );
        let topo_generator = TopologyGenerator::new(4, 1, (&model).into(), Some(topo_selector));
        let topologies = topo_generator.generate();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone(); 2];
        let particle_out = vec![model.get_particle_name("u").unwrap().clone(),
                                model.get_particle_name("u~").unwrap().clone()];
        let generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            1,
            model,
            Some(selector)
        );
        let diagrams = generator.assign_topology(&topologies[33]);
        println!("{}", &topologies[33]);
        assert_eq!(diagrams.len(), 4);
    }
}