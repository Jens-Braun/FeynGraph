use std::error::Error;
use itertools::Itertools;
use rayon::prelude::*;
use crate::model::TopologyModel;
use crate::topology::components::TopologyWorkspace;
use crate::topology::filter::TopologySelector;
use crate::topology::matrix::SymmetricMatrix;
use crate::util::{factorial, find_partitions};

pub mod matrix;
pub mod components;
pub mod filter;

#[derive(Debug, PartialEq)]
struct Topology {
    adjacency_matrix: SymmetricMatrix<usize>,
    node_symmetry: usize,
    edge_symmetry: usize,
}

impl Topology {
    pub fn from(workspace: &TopologyWorkspace, node_symmetry: usize) -> Self {
        let mut edge_symmetry = 1;
        for i in 0..workspace.adjacency_matrix.dimension {
            edge_symmetry *= 2_usize.pow((*workspace.adjacency_matrix.get(i, i)/2) as u32);
            edge_symmetry *= factorial(*workspace.adjacency_matrix.get(i, i)/2);
            for j in (i+1)..workspace.adjacency_matrix.dimension {
                edge_symmetry *= factorial(*workspace.adjacency_matrix.get(i, j));
            }
        }
        return Topology {
            adjacency_matrix: workspace.adjacency_matrix.clone(),
            node_symmetry,
            edge_symmetry
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct TopologyContainer {
    data: Vec<Topology>,
}

impl TopologyContainer {
    fn new() -> Self {
        return TopologyContainer {
            data: Vec::new(),
        }
    }
    
    fn with_capacity(capacity: usize) -> Self {
        return TopologyContainer {
            data: Vec::with_capacity(capacity)
        }
    }

    fn push(&mut self, topology: Topology) {
        self.data.push(topology);
    }

    fn inner_ref(&self) -> &Vec<Topology> {
        return &self.data;
    }

    fn inner_ref_mut(&mut self) -> &mut Vec<Topology> {
        return &mut self.data;
    }
}

impl From<Vec<TopologyContainer>> for TopologyContainer {
    fn from(containers: Vec<TopologyContainer>) -> Self {
        let mut result = TopologyContainer::with_capacity(
            containers.iter().map(|x| x.data.len()).sum()
        );
        for mut container in containers {
            result.inner_ref_mut().append(&mut container.data);
        }
        return result;
    }
}

pub struct TopologyGenerator {
    n_external: usize,
    n_loops: usize,
    model: TopologyModel,
    selector: TopologySelector,
}

impl TopologyGenerator {
    pub fn new(n_external: usize, n_loops: usize, model: TopologyModel, selector: Option<TopologySelector>) -> Self {
        return if let Some(selector) = selector {
            Self {
                n_external,
                n_loops,
                model,
                selector
            }
        } else {
            Self {
                n_external,
                n_loops,
                model,
                selector: TopologySelector::default()
            }
        }
    }
    
    pub fn generate(&self) -> Result<TopologyContainer, Box<dyn Error>> {
        // \sum_{k=3}^\infty (k-2) N_k = 2 L - 2 + E
        let node_partitions = find_partitions(
            self.model.degrees_iter().map(|d| d - 2), 
            2*self.n_loops + self.n_external - 2
        );
        let mut containers = Vec::new();
        node_partitions.into_par_iter().map(|partition| {
            let mut nodes = vec![1; self.n_external];
            let mut internal_nodes = partition
                .into_iter()
                .enumerate()
                .map(|(i, n)| vec![self.model.get(i); n])
                .concat();
            nodes.append(&mut internal_nodes);
            let mut workspace = TopologyWorkspace::from_nodes(&nodes);
            return workspace.generate();
        }).collect_into_vec(&mut containers);
        return Ok(TopologyContainer::from(containers));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::{prelude::*};
    
    #[test]
    fn topology_generator_1_loop_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let generator = TopologyGenerator::new(4, 1, model, None);
        let topologies = generator.generate().unwrap();
        assert_eq!(topologies.inner_ref().len(), 99);
    }

    #[test]
    fn topology_generator_3_loop_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let generator = TopologyGenerator::new(4, 3, model, None);
        let topologies = generator.generate().unwrap();
        assert_eq!(topologies.inner_ref().len(), 50051);
    }
}