use crate::topology::Topology;
use itertools::Itertools;

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
            edges,
        };
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AssignPropagator {
    pub particle: Option<usize>,
}

impl AssignPropagator {
    pub fn new() -> Self {
        return Self { particle: None };
    }
}

#[derive(Clone)]
pub(crate) struct VertexClassification {
    pub(crate) boundaries: Vec<usize>,
}

impl VertexClassification {
    pub(crate) fn get_class_sizes(&self) -> Vec<usize> {
        return self
            .boundaries
            .iter()
            .tuple_windows()
            .map(|(start, end)| *end - *start)
            .collect_vec();
    }

    pub(crate) fn get_class(&self, vertex: usize) -> usize {
        return self
            .boundaries
            .iter()
            .enumerate()
            .find_map(|(i, boundary)| if *boundary > vertex { Some(i) } else { None })
            .unwrap()
            - 1;
    }
    pub(crate) fn class_iter(&self, class: usize) -> impl Iterator<Item = usize> {
        return self.boundaries[class]..self.boundaries[class + 1];
    }
}

impl From<&Topology> for VertexClassification {
    fn from(topo: &Topology) -> Self {
        return Self {
            boundaries: topo.get_classification().boundaries.clone(),
        };
    }
}
