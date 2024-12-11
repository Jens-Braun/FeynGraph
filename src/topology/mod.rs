//! This module contains the [TopologyGenerator], which is the central object for generating
//! topologies of arbitrary node degrees, external particles and loops.

use std::cmp::min;
use std::fmt::Write;
use itertools::Itertools;
use rayon::prelude::*;
use crate::model::TopologyModel;
use crate::topology::workspace::TopologyWorkspace;
use crate::topology::filter::TopologySelector;
use crate::topology::matrix::SymmetricMatrix;
use crate::util::{factorial, find_partitions};

pub(crate) mod matrix;
pub(crate) mod components;
pub mod filter;
pub(crate) mod workspace;

#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    pub degree: usize,
    pub adjacent_nodes: Vec<usize>
}

impl Node {
    fn new(degree: usize, adjacent_nodes: Vec<usize>) -> Self {
        return Self {
            degree,
            adjacent_nodes
        }
    }
    
    fn from_matrix(degree: usize, matrix: &SymmetricMatrix<usize>, node_index: usize) -> Self {
        return Self {
            degree,
            adjacent_nodes: (0..matrix.dimension).filter(|i| *matrix.get(node_index, *i) > 0).collect_vec()
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Edge {
    pub connected_nodes: (usize, usize),
    pub momenta: Option<Vec<i8>>,
}

impl Edge {
    fn new(connected_nodes: (usize, usize), momenta: Option<Vec<i8>>) -> Self {
        return Self {
            connected_nodes,
            momenta
        }
    }
    
    fn empty(connected_nodes: (usize, usize)) -> Self {
        return Self {
            connected_nodes,
            momenta: None
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Topology {
    n_external: usize,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    node_symmetry: usize,
    edge_symmetry: usize,
    momentum_labels: Vec<String>,
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
        
        let n_momenta = workspace.n_external + workspace.n_loops;
        let nodes = workspace.nodes.iter().enumerate().map(|(i, topo_node)| Node::from_matrix(
            topo_node.max_connections,
            &workspace.adjacency_matrix,
            i
        )).collect_vec();
        let mut edges = Vec::new();
        for i in 0..workspace.nodes.len() {
            for _ in 0..*workspace.adjacency_matrix.get(i, i)/2 {
                edges.push(Edge::empty((i, i)));
            }
            for j in (i+1)..workspace.nodes.len() {
                for _ in 0..*workspace.adjacency_matrix.get(i, j) {
                    edges.push(Edge::empty((i, j)));
                }
            }
        }
        
        // Momentum Assignment
        let mut remaining_nodes = nodes.iter().map(|node| node.adjacent_nodes.len()).collect_vec();
        let mut current_loop_momentum: usize = workspace.n_external;
        
        // First assign loop momenta to self-loops
        for edge in edges.iter_mut().filter(|edge| edge.connected_nodes.0 == edge.connected_nodes.1) {
            let mut momenta = vec![0; n_momenta];
            momenta[current_loop_momentum] = 1;
            edge.momenta = Some(momenta);
            current_loop_momentum += 1;
            // Only decrease the number of remaining neighboring nodes if not done before, otherwise
            // the number will be decreased multiple times for nodes with multiple self-loops
            if nodes[edge.connected_nodes.0].adjacent_nodes.len() == remaining_nodes[edge.connected_nodes.0] {
                remaining_nodes[edge.connected_nodes.0] -= 1;
            }
        }
        
        // External momenta
        for edge in edges.iter_mut().take(workspace.n_external) {
            let mut momenta = vec![0; n_momenta];
            momenta[edge.connected_nodes.0] = 1;
            edge.momenta = Some(momenta);
            remaining_nodes[edge.connected_nodes.0] -= 1;
            remaining_nodes[edge.connected_nodes.1] -= 1;
        }
        
        // Remaining internal momenta
        while remaining_nodes.iter().any(|x| *x > 0) {
            // Always assign the node with the most momentum information
            let current_node = remaining_nodes.iter().position_min_by_key(
                |x| if **x > 0 {**x} else {usize::MAX}
            ).unwrap();
            
            // Momentum currently flowing into current_node
            let mut momenta = edges.iter()
                .filter(|edge| edge.momenta.is_some())
                .filter(|edge| edge.connected_nodes.0 == current_node || edge.connected_nodes.1 == current_node)
                .filter(|edge| edge.connected_nodes.0 != edge.connected_nodes.1)
                .fold(vec![0; n_momenta], |acc, edge| {
                    if edge.connected_nodes.1 == current_node {
                        acc.iter().zip((*edge.momenta.as_ref().unwrap()).iter()).map(|(x, y)| *x + *y).collect_vec()
                    } else {
                        acc.iter().zip((*edge.momenta.as_ref().unwrap()).iter()).map(|(x, y)| *x - *y).collect_vec()
                    }
                });
            // Next edge to which the momentum is assigned
            let mut current_edges = edges.iter_mut().filter(|edge|
                edge.momenta == None && (
                    edge.connected_nodes.0 == current_node ||
                        edge.connected_nodes.1 == current_node
                )
            ).collect_vec();
            let connected_node = if current_edges[0].connected_nodes.0 == current_node {
                current_edges[0].connected_nodes.1
            } else {
                current_edges[0].connected_nodes.0
            };
            current_edges = current_edges.into_iter().filter(
                |edge| {
                    (edge.connected_nodes.0 == current_node && edge.connected_nodes.1 == connected_node) ||
                        (edge.connected_nodes.1 == current_node && edge.connected_nodes.0 == connected_node)
                }
            ).collect_vec();
            
            // If there is more than one connection open, assign all at the same time
            if current_edges.len() > 1 {
                let n_edges = current_edges.len();
                momenta[current_loop_momentum] = 1;
                current_loop_momentum += 1;
                for (i, edge) in current_edges.into_iter().enumerate() {
                    // k_1 = p + l_1
                    if i == 0 {
                        edge.momenta = Some(momenta.clone());
                    } else {
                        // k_N = -l_{N-1}
                        if i == n_edges - 1 {
                            let mut momenta = vec![0; n_momenta];
                            momenta[current_loop_momentum - 1] = -1;
                            edge.momenta = Some(momenta);
                            remaining_nodes[edge.connected_nodes.0] -= 1;
                            remaining_nodes[edge.connected_nodes.1] -= 1;
                        } else {
                            // k_i = l_i - l_{i-1} for 1 < i < N
                            let mut momenta = vec![0; n_momenta];
                            momenta[current_loop_momentum - 1] = -1;
                            momenta[current_loop_momentum] = 1;
                            current_loop_momentum += 1;
                            edge.momenta = Some(momenta);
                        }
                    }
                }
            } else {
                current_edges[0].momenta = Some(momenta);
                remaining_nodes[current_edges[0].connected_nodes.0] -= 1;
                remaining_nodes[current_edges[0].connected_nodes.1] -= 1;
            }
        }
        
        // Use global momentum conservation to reduce momenta
        for edge in edges.iter_mut() {
            if edge.momenta.as_ref().unwrap().iter().take(workspace.n_external).all(|x| x.abs() == 1) {
                edge.momenta.as_mut().unwrap().iter_mut().take(workspace.n_external).for_each(|x| *x = 0);
            }
        }
        
        return Topology {
            n_external: workspace.n_external,
            nodes,
            edges,
            node_symmetry,
            edge_symmetry,
            momentum_labels: vec![
                (1..=workspace.n_external).map(|i| format!("p{}", i)).collect_vec(),
                (1..=workspace.n_loops).map(|i| format!("l{}", i)).collect_vec(),
            ].into_iter().flatten().collect_vec(),
        }
    }

    fn get_multiplicity(&self, first_node: usize, second_node: usize) -> usize {
        return self.edges.iter().filter(|edge| 
            edge.connected_nodes == (first_node, second_node) || edge.connected_nodes == (second_node, first_node)
        ).count();
    }
    
    #[allow(clippy::too_many_arguments)]
    fn bridge_dfs(&self, 
                  node: usize, 
                  parent: usize,
                  visited: &mut Vec<bool>, 
                  distance: &mut Vec<usize>, 
                  shortest_distance: &mut Vec<usize>, 
                  mut step: usize, 
                  bridges: &mut Vec<(usize, usize)>) {
        step += 1;
        distance[node] = step;
        shortest_distance[node] = step;
        visited[node] = true;
        for connected_node in self.nodes[node].adjacent_nodes.iter().cloned() {
            if connected_node == node { continue; }
            if connected_node < self.n_external || connected_node == parent {
                if self.get_multiplicity(node, connected_node) > 1 {
                    shortest_distance[node] = min(shortest_distance[node], shortest_distance[parent]);
                }
                continue;
            }
            if visited[connected_node] {
                shortest_distance[node] = min(shortest_distance[node], shortest_distance[connected_node]);
            } else {
                self.bridge_dfs(connected_node, node, visited, distance, shortest_distance, step, bridges);
                shortest_distance[node] = min(shortest_distance[node], shortest_distance[connected_node]);
                if shortest_distance[connected_node] > distance[node] {
                    bridges.push((node, connected_node));
                }
            }
        }
    }
    
    pub fn bridges(&self) -> Vec<(usize, usize)> {
        let mut bridges = Vec::new();
        let mut visited = vec![false; self.nodes.len()];
        let mut distance = vec![0; self.nodes.len()];
        let mut shortest_distance = vec![0; self.nodes.len()];
        let step = 0;
        self.bridge_dfs(self.n_external, 0, &mut visited, &mut distance, &mut shortest_distance, step, &mut bridges);
        return bridges;
    }
    
    pub fn count_opi(&self) -> usize {
        return self.bridges().len()+1;
    }
    
    pub fn nodes_iter(&self) -> impl Iterator<Item=&Node> {return self.nodes.iter(); }

    pub fn edges_iter(&self) -> impl Iterator<Item=&Edge> {return self.edges.iter(); }
    
    fn momentum_string(&self, edge_index: usize) -> String {
        let mut result = String::with_capacity(5*self.momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.edges[edge_index].momenta.as_ref().unwrap().iter().enumerate() {
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

impl std::fmt::Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Topology {{")?;
        write!(f, "    Nodes: [ ")?;
        for (i, node) in self.nodes.iter().enumerate() {
            if node.degree == 1 {
                write!(f, "N{}[ext] ", i)?;
            } else {
                write!(f, "N{}[{}] ", i, node.degree)?;
            }
        }
        writeln!(f, "]")?;
        writeln!(f, "    Edges: [")?;
        for (i, edge) in self.edges.iter().enumerate() {
            writeln!(f, "        {} -> {}, p = {},", 
                   edge.connected_nodes.0, 
                   edge.connected_nodes.1, 
                   self.momentum_string(i)
            )?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "    SymmetryFactor: 1/{}", self.node_symmetry * self.edge_symmetry)?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

/// Struct containing the topologies generated by a [TopologyGenerator].
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
    
    pub fn len(&self) -> usize {
        return self.data.len();
    }
    
    pub fn is_empty(&self) -> bool {
        return self.data.is_empty();
    }

    pub fn get(&self, i: usize) -> &Topology {
        return &self.data[i];
    }
    
    /// Search for topologies which would be selected by `selector`. Returns the index of the first selected diagram
    /// or `None` if no diagram is selected.
    pub fn query(&self, selector: &TopologySelector) -> Option<usize> {
        return if let Some((i, _)) = self.data.iter().find_position(|topo| selector.select(topo)) {
            Some(i)
        } else {
            None
        }
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

/// A generator to construct all possible topologies given by
/// 
/// - a [TopologyModel] defining the possible degrees of appearing nodes
/// - `n_external` external particles
/// - `n_loops` loops
/// - a [TopologySelector] deciding which diagrams are discarded during the generation
/// 
/// # Examples
/// ```rust
/// use feyngraph::model::TopologyModel;
/// use feyngraph::topology::{TopologyGenerator, filter::TopologySelector};
/// 
/// // Use vertices with degree 3 and 4 for the topologies
/// let model = TopologyModel::from(vec![3, 4]);
/// 
/// // Construct only one-particle-irreducible (one 1PI-component) diagrams
/// let mut selector = TopologySelector::default();
/// selector.add_opi_count(1);
/// 
/// // Generate all three-point topologies with three loops with the given model and selector
/// let generator = TopologyGenerator::new(3, 3, model, Some(selector));
/// let topologies = generator.generate();
/// 
/// assert_eq!(topologies.len(), 619);
/// ```
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
    
    /// Generate the topologies.
    pub fn generate(&self) -> TopologyContainer {
        let degrees = self.model.degrees_iter().collect_vec();
        // Let N_k be the number of nodes with degree k, then
        //      \sum_{k=3}^\infty (k-2) N_k = 2 L - 2 + E                                        (1)
        // where L is the number of loops and E is the number of external particles.
        // The full set of diagrams is then the sum of all node partitions {N_k}, such that (1) is satisfied.
        let node_partitions = find_partitions(
            self.model.degrees_iter().map(|d| d - 2), 2*self.n_loops + self.n_external - 2
        ).into_iter()
            .filter(|partition| {
                self.selector.select_partition(
                    partition.iter().enumerate().map(|(i, count)| (degrees[i], *count)).collect_vec()
                )
            }
        ).collect_vec();
        
        let mut containers = Vec::new();
        node_partitions
            .into_par_iter()
            .map(|partition| {
            let mut nodes = vec![1; self.n_external];
            let mut internal_nodes = partition
                .into_iter()
                .enumerate()
                .map(|(i, n)| vec![self.model.get(i); n])
                .concat();
            nodes.append(&mut internal_nodes);
            let mut workspace = TopologyWorkspace::from_nodes(self.n_external, self.n_loops, &nodes);
            workspace.topology_selector = self.selector.clone();
            return workspace.generate();
        }).collect_into_vec(&mut containers);
        return TopologyContainer::from(containers);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use crate::topology::filter::SelectionCriterion::CustomCriterion;
    use super::*;
    
    #[test]
    fn topology_generator_custom_function_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let mut selector = TopologySelector::default();
        let filter = |topo: &Topology| -> bool {
            for edge in topo.edges.iter() {
                if edge.connected_nodes.0 == edge.connected_nodes.1 {
                    return false;
                }
            }
            return true;
        };
        selector.add_criterion(CustomCriterion(Arc::new(filter)));
        let generator = TopologyGenerator::new(2, 1, model, Some(selector));
        let topologies = generator.generate();
        assert_eq!(topologies.len(), 1);
    }
    
    #[test]
    fn topology_generator_1_loop_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let generator = TopologyGenerator::new(4, 1, model, Some(TopologySelector::new()));
        let topologies = generator.generate();
        assert_eq!(topologies.inner_ref().len(), 99);
    }

    #[test]
    fn topology_generator_3_loop_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let generator = TopologyGenerator::new(4, 3, model, Some(TopologySelector::new()));
        let topologies = generator.generate();
        assert_eq!(topologies.inner_ref().len(), 50051);
    }
    
    #[test]
    fn topology_generator_3_loop_opi_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let selector = TopologySelector::default();
        let generator = TopologyGenerator::new(4, 3, model, Some(selector));
        let topologies = generator.generate();
        assert_eq!(topologies.inner_ref().len(), 6166);
    }

    #[test]
    fn topology_generator_3_loop_opi_partition_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let mut selector = TopologySelector::default();
        selector.add_node_partition(vec![(3, 4), (4, 2)]);
        let generator = TopologyGenerator::new(4, 3, model, Some(selector));
        let topologies = generator.generate();
        assert_eq!(topologies.inner_ref().len(), 2614);
    }

    #[test]
    fn topology_generator_2_loop_degree_6_test() {
        let model = TopologyModel::from(vec![3, 4, 5, 6]);
        let selector = TopologySelector::default();
        let generator = TopologyGenerator::new(4, 2, model, Some(selector));
        let topologies = generator.generate();
        assert_eq!(topologies.inner_ref().len(), 404);
    }
    
    #[test]
    fn topology_bridge_test() {
        let adjacency_matrix = SymmetricMatrix::from_vec(10,
                                                         vec![
                                                             0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                                0, 0, 0, 0, 0, 1, 0, 0, 0,
                                                                   0, 0, 0, 0, 0, 0, 1, 0,
                                                                      0, 0, 0, 0, 0, 0, 1,
                                                                         0, 1, 0, 1, 0, 0,
                                                                            0, 1, 1, 0, 0,
                                                                               2, 0, 0, 0,
                                                                                  0, 2, 0,
                                                                                     0, 1,
                                                                                        2
                                                         ]);
        let degrees = vec![1, 1, 1, 1, 3, 3, 4, 4, 4, 4];
        let mut edges:Vec<Edge> = Vec::new();
        for i in 0..10 {
            for _ in 0..*adjacency_matrix.get(i, i)/2 {
                edges.push(Edge::empty((i, i)))
            }
            for j in (i+1)..10 {
                for _ in 0..*adjacency_matrix.get(i, j) {
                    edges.push(Edge::empty((i, j)))
                }
            }
        }
        let topo = Topology {
            n_external: 4,
            nodes: (0..10).map(|i| Node::from_matrix(degrees[i], &adjacency_matrix, i)).collect_vec(),
            edges,
            node_symmetry: 1,
            edge_symmetry: 1,
            momentum_labels: vec!["p1", "p2", "p3", "p4", "l1", "l2", "l3", "l4"]
                .into_iter().map(|x| x.to_string()).collect_vec(),
        };
        println!("{:#?}", topo);
        assert_eq!(topo.bridges().into_iter().collect::<HashSet<(usize, usize)>>(), 
                   HashSet::from([(5, 6), (8, 9)]));
    }
}