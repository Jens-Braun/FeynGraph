use itertools::Itertools;
use crate::model::Particle;
use crate::topology::{Edge, Topology};

#[derive(Debug, Clone, PartialEq)]
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
    
    pub fn assign_particle(edge: &Edge, particle: &Particle) -> Self {
        return Self {
            vertices: edge.connected_nodes.clone(),
            particle: particle.clone(),
            momentum: edge.momenta.clone().unwrap()
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AssignVertex {
    pub degree: usize,
    pub remaining_legs: usize,
    pub candidates: Vec<usize>,
    pub edges: Vec<usize>,
}

impl AssignVertex {
    pub(crate) fn new(degree: usize, edges: Vec<usize>) -> Self {
        return Self {
            degree,
            remaining_legs: degree,
            candidates: Vec::new(),
            edges
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AssignPropagator {
    pub particle: Option<usize>
}

impl AssignPropagator {
    pub fn new() -> Self {
        return Self { particle: None };
    }
}

#[derive(Clone)]
pub(crate) struct VertexClassification {
    pub(crate) boundaries: Vec<usize>,
    pub(crate) matrix: Vec<Vec<usize>>
}

impl VertexClassification {
    pub(crate) fn get_class_sizes(&self) -> Vec<usize> {
        return self.boundaries.iter().tuple_windows().map(|(start, end)| *end - *start).collect_vec();
    }

    pub(crate) fn get_class(&self, vertex: usize) -> usize {
        return self.boundaries.iter().enumerate().find_map(
            |(i, boundary)| if *boundary > vertex {Some(i)} else {None}
        ).unwrap().clone()-1;
    }
    pub(crate) fn class_iter(&self, class: usize) -> impl Iterator<Item = usize> {
        return self.boundaries[class]..self.boundaries[class+1];
    }

    pub(crate) fn update_classification(&mut self, topology: &Topology, vertices: &Vec<AssignVertex>) {
        let mut changed = true;
        let mut new_boundaries = Vec::new();
        while changed {
            changed = false;
            for (i, (start, end)) in self.boundaries.iter().tuple_windows().enumerate() {
                if *end == (*start + 1) { continue; }
                for vertex in *start..(*end - 1) {
                    if vertices[vertex].candidates.len() == 1
                        && vertices[vertex + 1].candidates.len() == 1
                        && vertices[vertex].candidates[0] != vertices[vertex + 1].candidates[0] {
                        new_boundaries.push((i+1, vertex+1));
                        changed = true;
                    }
                }
            }
            for (i, boundary) in std::mem::take(&mut new_boundaries).into_iter() {
                self.boundaries.insert(i, boundary);
            }
            if changed {
                self.update_classification_matrix(topology);
            }
        }
    }

    fn update_classification_matrix(&mut self, topology: &Topology) {
        let n_classes = self.boundaries.len()-1;
        for row in &mut self.matrix {
            *row = vec![0; n_classes];
        }
        for edge in topology.edges.iter() {
            let initial_class = self.get_class(edge.connected_nodes.0);
            let final_class = self.get_class(edge.connected_nodes.1);
            self.matrix[edge.connected_nodes.0][final_class] += 1;
            self.matrix[edge.connected_nodes.1][initial_class] += 1;
        }
    }
}

impl From<&Topology> for VertexClassification {
    fn from(topo: &Topology) -> Self {
        return Self {
            boundaries: topo.get_classification().boundaries.clone(),
            matrix: topo.get_classification().matrix.clone()
        }
    }
}