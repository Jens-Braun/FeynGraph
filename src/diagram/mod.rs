use std::ops::Deref;
use std::fmt::Write;
use itertools::Itertools;
use crate::diagram::components::Propagator;
use crate::diagram::filter::DiagramSelector;
use crate::model::{Model, Particle, TopologyModel, Vertex};
use crate::topology::filter::TopologySelector;
use crate::topology::{Topology, TopologyGenerator};

use rayon::prelude::*;
use crate::diagram::workspace::AssignWorkspace;

mod components;
pub mod filter;
mod workspace;

#[derive(Debug, PartialEq)]
pub struct Diagram {
    incoming_particles: Vec<Particle>,
    outgoing_particles:Vec<Particle>,
    vertices: Vec<Vertex>,
    propagators: Vec<Propagator>,
    vertex_symmetry: usize,
    propagator_symmetry: usize,
    momentum_labels: Vec<String>,
}

impl Diagram {
    fn from(workspace: &AssignWorkspace) -> Self {
        return Diagram {
            incoming_particles: workspace.incoming_particles.clone(),
            outgoing_particles: workspace.outgoing_particles.clone(),
            vertices: workspace.vertex_candidates.iter()
                .filter_map(|candidate| {
                    if candidate.degree == 1 {
                        None
                    } else {
                        Some(workspace.model.get_vertex(candidate.candidates[0]).clone())
                    }
                }).collect_vec(),
            propagators: workspace.propagator_candidates.iter().enumerate()
                .map(|(i, candidate)|
                    Propagator::new(
                        workspace.topology.get_edge(i).connected_nodes,
                        workspace.model.get_particle(candidate.particle.unwrap()).clone(),
                        workspace.topology.get_edge(i).momenta.as_ref().unwrap().clone(),
                    )
                ).collect_vec(),
            vertex_symmetry: workspace.topology.get_node_symmetry(),
            propagator_symmetry: workspace.topology.get_edge_symmetry(),
            momentum_labels: workspace.topology.momentum_labels.clone()
        };
    }

    fn momentum_string(&self, edge_index: usize) -> String {
        let mut result = String::with_capacity(5*self.momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.propagators[edge_index].momentum.iter().enumerate() {
            if *coefficient == 0 { continue; }
            if first {
                write!(&mut result, "{}*{} ", coefficient, self.momentum_labels[i]).unwrap();
                first = false;
            } else {
                match coefficient.signum() {
                    1 => {
                        write!(&mut result, "+ {}*{} ", coefficient, self.momentum_labels[i]).unwrap();
                    },
                    -1 => {
                        write!(&mut result, "- {}*{} ", coefficient.abs(), self.momentum_labels[i]).unwrap();
                    },
                    _ => unreachable!()
                }
            }
        }
        return result;
    }
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
                     self.momentum_string(i)
            )?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "    SymmetryFactor: 1/{}", self.vertex_symmetry * self.propagator_symmetry)?;
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
            n_loops
        }
    }
    
    pub fn generate(&mut self) -> DiagramContainer {
        let mut topo_generator = TopologyGenerator::new(
            self.n_external,
            self.n_loops,
            TopologyModel::from(&self.model),
            Some(TopologySelector::from(&self.selector)),
        );
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
        let counts = containers
            .iter()
            .map(|container| container.len())
            .counts()
            .into_iter()
            .sorted()
            .collect_vec();
        println!("{:#?}", counts);
        return DiagramContainer::from(containers);
    }

    pub fn assign_topology(&mut self, topology: &Topology) -> DiagramContainer {
        let mut assign_workspace = AssignWorkspace::new(
            topology,
            &self.model,
            &self.selector,
            &self.incoming_particles,
            &self.outgoing_particles
        );
        return assign_workspace.assign();
    }

    pub fn assign_topologies(&mut self, topologies: &[Topology]) -> DiagramContainer {
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
        let counts = containers
            .iter()
            .map(|container| container.len())
            .counts()
            .into_iter()
            .sorted()
            .collect_vec();
        println!("{:#?}", counts);
        return DiagramContainer::from(containers);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use crate::topology::filter::SelectionCriterion::CustomCriterion;
    use super::*;
    
    #[test]
    pub fn diagram_generator_qcd_2g_2g_tree() {
        let model = Model::from_ufo(&PathBuf::from("tests/QCD_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone(); 2];
        let particle_out = particles_in.clone();
        let mut generator = DiagramGenerator::new(
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
            model.get_particle_name("u__tilde__").unwrap().clone()
        ];
        let mut generator = DiagramGenerator::new(
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
            model.get_particle_name("u__tilde__").unwrap().clone()
        ];
        let mut generator = DiagramGenerator::new(
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
            model.get_particle_name("u__tilde__").unwrap().clone()
        ];
        let mut generator = DiagramGenerator::new(
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
            model.get_particle_name("u__tilde__").unwrap().clone()
        ];
        let mut generator = DiagramGenerator::new(
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
        let mut generator = DiagramGenerator::new(
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
        let mut generator = DiagramGenerator::new(
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
        let mut generator = DiagramGenerator::new(
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
        selector.set_opi();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("G").unwrap().clone()];
        let mut generator = DiagramGenerator::new(
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
        topo_selector.add_criterion(CustomCriterion(
            Arc::new(|topo: &Topology| -> bool {
                !topo.edges_iter().any(|edge| edge.connected_nodes.0 == edge.connected_nodes.1)
            })
        ));
        let topo_generator = TopologyGenerator::new(2, 3, (&model).into(), Some(topo_selector));
        let topologies = topo_generator.generate();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_name("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_name("G").unwrap().clone()];
        let mut generator = DiagramGenerator::new(
            particles_in,
            particle_out,
            3,
            model,
            Some(selector)
        );
        let diagrams = generator.assign_topologies(&topologies);
        assert_eq!(diagrams.len(), 951);
    }
}