//! Central module for the generation of unassigned graphs, called _topologies_.
//!
//! The central object of this module is the [`TopologyGenerator`], which handles the generation
//! of the topologies given a [`TopologyModel`] and optionally a [`TopologySelector`]
//! restricting which topologies are generated.

pub use crate::model::TopologyModel;
use crate::util::{Error, factorial, find_partitions};
use components::NodeClassification;
pub use filter::TopologySelector;
use itertools::Itertools;
use matrix::SymmetricMatrix;
use rayon::prelude::*;
use std::cmp::min;
use std::fmt::Write;
use std::ops::{Deref, Index};
use workspace::TopologyWorkspace;

pub(crate) mod components;
pub(crate) mod filter;
pub(crate) mod matrix;
pub(crate) mod workspace;

/// A struct representing a topological node.
#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    /// The degree of the node, i.e. the number of legs connected.
    pub degree: usize,
    /// List of identifiers of nodes which are adjacent to this one.
    pub adjacent_nodes: Vec<usize>,
}

impl Node {
    fn new(degree: usize, adjacent_nodes: Vec<usize>) -> Self {
        return Self { degree, adjacent_nodes };
    }

    fn from_matrix(degree: usize, matrix: &SymmetricMatrix, node_index: usize) -> Self {
        return Self {
            degree,
            adjacent_nodes: (0..matrix.dimension)
                .filter(|i| *matrix.get(node_index, *i) > 0)
                .collect_vec(),
        };
    }
}

/// A struct representing a topological edge.
#[derive(Debug, PartialEq, Clone)]
pub struct Edge {
    /// Identifiers of the nodes this edge is attached to. Always ordered such that `connected_nodes[0] <= connected_nodes[1]`.
    pub connected_nodes: [usize; 2],
    /// Internal representation of the node's momentum. The `i`-th entry in the list is the coefficient of the `i`-th momentum,
    /// where the first `n_external` momenta are the external ones and the remaining ´n_loops` momenta are the loop-momenta.
    pub momenta: Option<Vec<i8>>,
}

impl Edge {
    fn new(connected_nodes: [usize; 2], momenta: Option<Vec<i8>>) -> Self {
        return Self {
            connected_nodes,
            momenta,
        };
    }

    fn empty(connected_nodes: [usize; 2]) -> Self {
        return Self {
            connected_nodes,
            momenta: None,
        };
    }
}

/// An undirected graph with a given number of loops and external legs.
#[derive(Debug, PartialEq, Clone)]
pub struct Topology {
    pub(crate) n_external: usize,
    pub(crate) n_loops: usize,
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) node_symmetry: usize,
    pub(crate) edge_symmetry: usize,
    pub(crate) momentum_labels: Vec<String>,
    pub(crate) bridges: Vec<(usize, usize)>,
    pub(crate) node_classification: NodeClassification,
}

impl Topology {
    fn from(workspace: &TopologyWorkspace, node_symmetry: usize) -> Self {
        let mut edge_symmetry = 1;
        for i in 0..workspace.adjacency_matrix.dimension {
            edge_symmetry *= 2_usize.pow((*workspace.adjacency_matrix.get(i, i) / 2) as u32);
            edge_symmetry *= factorial(*workspace.adjacency_matrix.get(i, i) / 2);
            for j in (i + 1)..workspace.adjacency_matrix.dimension {
                edge_symmetry *= factorial(*workspace.adjacency_matrix.get(i, j));
            }
        }

        let nodes = workspace
            .nodes
            .iter()
            .enumerate()
            .map(|(i, topo_node)| Node::from_matrix(topo_node.max_connections, &workspace.adjacency_matrix, i))
            .collect_vec();
        let mut edges = Vec::new();
        for i in 0..workspace.nodes.len() {
            for _ in 0..*workspace.adjacency_matrix.get(i, i) / 2 {
                edges.push(Edge::empty([i, i]));
            }
            for j in (i + 1)..workspace.nodes.len() {
                for _ in 0..*workspace.adjacency_matrix.get(i, j) {
                    edges.push(Edge::empty([i, j]));
                }
            }
        }
        let mut topo = Topology {
            n_external: workspace.n_external,
            n_loops: workspace.n_loops,
            nodes,
            edges,
            node_symmetry,
            edge_symmetry,
            momentum_labels: workspace.momentum_labels.clone(),
            bridges: Vec::new(),
            node_classification: workspace.node_classification.clone(),
        };
        topo.assign_momenta();
        return topo;
    }

    /// Assign momenta to the edges and find the bridges of the topology.
    fn assign_momenta(&mut self) {
        let mut current_loop_momentum = self.n_external;
        let n_momenta = self.n_external + self.n_loops;

        // First assign loop momenta to self-loops
        for edge in self
            .edges
            .iter_mut()
            .filter(|edge| edge.connected_nodes[0] == edge.connected_nodes[1])
        {
            let mut momenta = vec![0; n_momenta];
            momenta[current_loop_momentum] = 1;
            edge.momenta = Some(momenta);
            current_loop_momentum += 1;
        }

        let mut visited = vec![false; self.nodes.len()];
        let mut distance = vec![0; self.nodes.len()];
        let mut shortest_distance = vec![0; self.nodes.len()];
        let mut momentum_distance = vec![Vec::new(); self.nodes.len()];
        let mut momenta = vec![0; self.momentum_labels.len()];

        let step = 0;
        // Traverse the topology in depth-first order. The algorithm is a modified version of Tarjan's
        // bridge-finding algorithm, which keeps track of which loops the current edge is part of.
        self.momentum_dfs(
            self.nodes[0].adjacent_nodes[0],
            0,
            &mut visited,
            &mut distance,
            &mut shortest_distance,
            &mut momentum_distance,
            step,
            &mut current_loop_momentum,
            &mut momenta,
        );

        // External momenta
        for edge in self.edges.iter_mut().take(self.n_external) {
            let mut momenta = vec![0; n_momenta];
            momenta[edge.connected_nodes[0]] = 1;
            edge.momenta = Some(momenta);
        }

        // Use global momentum conservation to eliminate the last external momentum
        if self.n_external > 1 {
            for edge in self.edges.iter_mut() {
                if !(self.nodes[edge.connected_nodes[0]].degree == 1 || self.nodes[edge.connected_nodes[1]].degree == 1)
                    && edge.momenta.as_ref().unwrap()[self.n_external - 1] != 0
                {
                    for i in 0..self.n_external {
                        edge.momenta.as_mut().unwrap()[i] -= edge.momenta.as_ref().unwrap()[self.n_external - 1];
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn momentum_dfs(
        &mut self,
        node: usize,
        parent: usize,
        visited: &mut Vec<bool>,
        distance: &mut Vec<usize>,
        shortest_distance: &mut Vec<usize>,
        momentum_distance: &mut Vec<Vec<usize>>,
        mut step: usize,
        current_loop_momentum: &mut usize,
        momenta: &mut Vec<i8>,
    ) {
        step += 1;
        // Mark the current node as visited and initialize the distances
        distance[node] = step;
        shortest_distance[node] = step;
        momentum_distance[node] = vec![step; self.n_external + self.n_loops];
        visited[node] = true;
        let mut local_momenta = vec![0; momenta.len()];
        for connected_node in self.nodes[node].adjacent_nodes.clone().into_iter() {
            // External node -> accumulate the momentum for the edge on the way back, ignore otherwise
            if connected_node < self.n_external {
                local_momenta[connected_node] = -1;
                continue;
            }
            // Ignore self-loops and edges to which a momentum is already assigned
            if connected_node == node
                || self
                    .edges
                    .iter()
                    .filter(|edge| {
                        edge.connected_nodes == [node, connected_node] || edge.connected_nodes == [connected_node, node]
                    })
                    .any(|edge| edge.momenta.is_some())
            {
                continue;
            }
            // Parent node -> if there are multiple connections between the current node and the parent node,
            // the current edge cannot be a bridge, ignore otherwise
            if connected_node == parent {
                if self.get_multiplicity(node, connected_node) > 1 {
                    shortest_distance[node] = min(shortest_distance[node], shortest_distance[parent]);
                }
                continue;
            }
            // Invert momentum, because the current direction is opposite to the momentum flow
            let invert_direction = node > connected_node;

            if visited[connected_node] {
                // Node has already been visited -> found a new loop
                shortest_distance[node] = min(shortest_distance[node], shortest_distance[connected_node]);
                // Assign a new loop momentum to the current edge (node, connected_node)
                // All nodes on the way back are assigned the dfs-depth of connected_node until
                // connected_node is reached again, which closes the loop
                momentum_distance[node][*current_loop_momentum] = min(
                    momentum_distance[node][*current_loop_momentum],
                    momentum_distance[connected_node][*current_loop_momentum],
                );
                let mut momentum = vec![0; self.n_external + self.n_loops];
                momentum[*current_loop_momentum] = 1;
                *current_loop_momentum += 1;
                if !invert_direction {
                    self.assign_momentum(node, connected_node, momentum, current_loop_momentum);
                } else {
                    self.assign_momentum(
                        node,
                        connected_node,
                        momentum.clone().into_iter().map(|x| -x).collect(),
                        current_loop_momentum,
                    );
                }
            } else {
                // Unknown node -> go deeper
                let mut accumulated_momenta = vec![0; momenta.len()];
                self.momentum_dfs(
                    connected_node,
                    node,
                    visited,
                    distance,
                    shortest_distance,
                    momentum_distance,
                    step,
                    current_loop_momentum,
                    &mut accumulated_momenta,
                );
                shortest_distance[node] = min(shortest_distance[node], shortest_distance[connected_node]);

                // All external momenta picked up on the way back up to the current node
                let mut current_momentum = accumulated_momenta.clone();

                // Add to total momentum flow of the current node
                *momenta = momenta
                    .iter_mut()
                    .zip(accumulated_momenta)
                    .map(|(x, y)| *x + y)
                    .collect();
                for l in self.n_external..(self.n_external + self.n_loops) {
                    // Check if the current edge is part of the l-th loop
                    // The distance of connected_node is the one of the node which initiated the loop,
                    // therefore the loop is not closed yet if the distance of node is different
                    if momentum_distance[connected_node][l] <= momentum_distance[node][l] {
                        current_momentum[l] = 1;
                    }
                    momentum_distance[node][l] = min(momentum_distance[node][l], momentum_distance[connected_node][l]);
                }

                if !invert_direction {
                    self.assign_momentum(node, connected_node, current_momentum, current_loop_momentum);
                } else {
                    self.assign_momentum(
                        node,
                        connected_node,
                        current_momentum.clone().into_iter().map(|x| -x).collect(),
                        current_loop_momentum,
                    );
                }

                if shortest_distance[connected_node] > distance[node] {
                    self.bridges.push((node, connected_node));
                }
            }
        }
        // Accumulate all external momenta flowing into the current node for the remaining way back
        *momenta = momenta.iter_mut().zip(local_momenta).map(|(x, y)| *x + y).collect();
    }

    /// Assign `momentum` to the edges connecting `first_node` and `second_node`. If there are
    /// multiple edges connecting the nodes, additional loop momenta are assigned.
    fn assign_momentum(
        &mut self,
        first_node: usize,
        second_node: usize,
        momentum: Vec<i8>,
        current_loop_momentum: &mut usize,
    ) {
        let n_momenta = self.n_external + self.n_loops;
        let current_edges = self
            .edges
            .iter_mut()
            .filter(|other_edge| {
                other_edge.connected_nodes == [first_node, second_node]
                    || other_edge.connected_nodes == [second_node, first_node]
            })
            .collect_vec();
        let n_edges = current_edges.len();
        for (i, edge) in current_edges.into_iter().enumerate() {
            // k_1 = p + l_1
            if i == 0 {
                let mut momentum = momentum.clone();
                if n_edges > 1 {
                    momentum[*current_loop_momentum] = 1;
                    *current_loop_momentum += 1;
                }
                edge.momenta = Some(momentum);
            } else {
                // k_N = -l_{N-1}
                if i == n_edges - 1 {
                    let mut momenta = vec![0; n_momenta];
                    momenta[*current_loop_momentum - 1] = -1;
                    edge.momenta = Some(momenta);
                } else {
                    // k_i = l_i - l_{i-1} for 1 < i < N
                    let mut momenta = vec![0; n_momenta];
                    momenta[*current_loop_momentum - 1] = -1;
                    momenta[*current_loop_momentum] = 1;
                    *current_loop_momentum += 1;
                    edge.momenta = Some(momenta);
                }
            }
        }
    }

    fn get_multiplicity(&self, first_node: usize, second_node: usize) -> usize {
        return self
            .edges
            .iter()
            .filter(|edge| {
                edge.connected_nodes == [first_node, second_node] || edge.connected_nodes == [second_node, first_node]
            })
            .count();
    }

    /// Return the bridges of the topology. A bridge is an edge that would make the graph disconnected when cut.
    pub fn bridges(&self) -> Vec<(usize, usize)> {
        return self.bridges.clone();
    }

    /// Count the number of one-particle-irreducible components of the topology.
    pub fn count_opi_componenets(&self) -> usize {
        return self.bridges().len() + 1;
    }

    /// Count the number of self-loops in the topology. A self-loop is defined as an edge which ends on the same
    /// node it started on.
    pub fn count_self_loops(&self) -> usize {
        return self
            .edges
            .iter()
            .filter(|edge| edge.connected_nodes[0] == edge.connected_nodes[1])
            .count();
    }

    /// Return `true` if the topology is on-shell, i.e. contains no self-energy insertions on an external edge. This
    /// implementation considers internal edges carrying a single external momentum and no loop momentum, which is
    /// equivalent to a self-energy insertion on an external edge.
    pub fn on_shell(&self) -> bool {
        return !self.edges.iter().any(|edge| {
            return if edge.connected_nodes[0] >= self.n_external && edge.connected_nodes[1] >= self.n_external && // internal edge
                edge.momenta.as_ref().unwrap().iter().skip(self.n_external).all(|x| *x == 0)
            {
                // no loop momentum
                let ext_count = edge
                    .momenta
                    .as_ref()
                    .unwrap()
                    .iter()
                    .take(self.n_external)
                    .filter(|x| **x != 0)
                    .count();
                match ext_count {
                    1 => true,
                    n if n + 1 == self.n_external => {
                        // Try with momentum conservation
                        edge.momenta
                            .as_ref()
                            .unwrap()
                            .iter()
                            .take(self.n_external)
                            .map(|x| x.unsigned_abs() as usize)
                            .sum::<usize>()
                            + 1
                            == self.n_external
                    }
                    _ => false,
                }
            } else {
                false
            };
        });
    }

    /// Return a reference to the topology's `i`-th node.
    pub fn get_node(&self, i: usize) -> &Node {
        return &self.nodes[i];
    }

    /// Return an iterator over the topology's edges.
    pub fn nodes_iter(&self) -> impl Iterator<Item = &Node> {
        return self.nodes.iter();
    }

    /// Return a reference to the `i`-th edge.
    pub fn get_edge(&self, i: usize) -> &Edge {
        return &self.edges[i];
    }

    /// Return an iterator over the topology's edges.
    pub fn edges_iter(&self) -> impl Iterator<Item = &Edge> {
        return self.edges.iter();
    }

    /// Return the ids of the edges connected to node `i`, including `i` if it is connected to itself via a self-loop.
    pub fn adjacent_nodes(&self, i: usize) -> &Vec<usize> {
        return &self.nodes[i].adjacent_nodes;
    }

    /// Return the number of nodes.
    pub fn n_nodes(&self) -> usize {
        return self.nodes.len();
    }

    /// Return the number of edges.
    pub fn n_edges(&self) -> usize {
        return self.edges.len();
    }

    pub(crate) fn get_classification(&self) -> &NodeClassification {
        return &self.node_classification;
    }

    pub(crate) fn get_node_symmetry(&self) -> usize {
        return self.node_symmetry;
    }

    pub(crate) fn get_edge_symmetry(&self) -> usize {
        return self.edge_symmetry;
    }

    /// Get the momentum of edge `edge_index` as a formatted string.
    pub fn momentum_string(&self, edge_index: usize) -> String {
        let mut result = String::with_capacity(5 * self.momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.edges[edge_index].momenta.as_ref().unwrap().iter().enumerate() {
            if *coefficient == 0 {
                continue;
            }
            if first {
                write!(&mut result, "{}*{} ", coefficient, self.momentum_labels[i]).unwrap();
                first = false;
            } else {
                match coefficient.signum() {
                    1 => {
                        write!(&mut result, "+ {}*{} ", coefficient, self.momentum_labels[i]).unwrap();
                    }
                    -1 => {
                        write!(&mut result, "- {}*{} ", coefficient.abs(), self.momentum_labels[i]).unwrap();
                    }
                    _ => unreachable!(),
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
            writeln!(
                f,
                "        {} -> {}, p = {},",
                edge.connected_nodes[0],
                edge.connected_nodes[1],
                self.momentum_string(i)
            )?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "    SymmetryFactor: 1/{}", self.node_symmetry * self.edge_symmetry)?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

/// Smart container for the topologies generated by a [TopologyGenerator].
#[derive(Debug, PartialEq)]
pub struct TopologyContainer {
    pub(crate) data: Vec<Topology>,
}

impl TopologyContainer {
    fn new() -> Self {
        return TopologyContainer { data: Vec::new() };
    }

    fn with_capacity(capacity: usize) -> Self {
        return TopologyContainer {
            data: Vec::with_capacity(capacity),
        };
    }

    fn push(&mut self, topology: Topology) {
        self.data.push(topology);
    }

    pub(crate) fn inner_ref(&self) -> &Vec<Topology> {
        return &self.data;
    }

    fn inner_ref_mut(&mut self) -> &mut Vec<Topology> {
        return &mut self.data;
    }

    /// Return the number of topologies in the container.
    pub fn len(&self) -> usize {
        return self.data.len();
    }

    /// Check whether the container is empty or not.
    pub fn is_empty(&self) -> bool {
        return self.data.is_empty();
    }

    /// Return a reference to the `i`-th diagram in the container.
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
        };
    }
}

impl From<Vec<TopologyContainer>> for TopologyContainer {
    fn from(containers: Vec<TopologyContainer>) -> Self {
        let mut result = TopologyContainer::with_capacity(containers.iter().map(|x| x.data.len()).sum());
        for mut container in containers {
            result.inner_ref_mut().append(&mut container.data);
        }
        return result;
    }
}

impl Index<usize> for TopologyContainer {
    type Output = Topology;
    fn index(&self, i: usize) -> &Self::Output {
        return &self.data[i];
    }
}

impl Deref for TopologyContainer {
    type Target = Vec<Topology>;

    fn deref(&self) -> &Self::Target {
        return &self.data;
    }
}

/// The central object of the topology module, generating all possible topologies for a given problem.
///
/// The problem is specified by
///
/// - a [TopologyModel] defining the possible degrees of appearing nodes
/// - `n_external` external particles
/// - `n_loops` loops
/// - a [TopologySelector] deciding which diagrams are discarded during the generation
///
/// # Examples
/// ```rust
/// use feyngraph::model::TopologyModel;
/// use feyngraph::topology::{TopologyGenerator, TopologySelector};
///
/// // Use vertices with degree 3 and 4 for the topologies
/// let model = TopologyModel::from(vec![3, 4]);
///
/// // Construct only one-particle-irreducible (one 1PI-component) diagrams
/// let mut selector = TopologySelector::default();
/// selector.select_opi_components(1);
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
    momentum_labels: Option<Vec<String>>,
}

impl TopologyGenerator {
    /// Create a new topology generator for `n_external` external legs, `n_loops` loops within `model`. Whether a
    /// topology is kept or discarded during generation is determined by `selector`.
    pub fn new(n_external: usize, n_loops: usize, model: TopologyModel, selector: Option<TopologySelector>) -> Self {
        return if let Some(selector) = selector {
            Self {
                n_external,
                n_loops,
                model,
                selector,
                momentum_labels: None,
            }
        } else {
            Self {
                n_external,
                n_loops,
                model,
                selector: TopologySelector::default(),
                momentum_labels: None,
            }
        };
    }

    /// Set the names of the momenta. The first `n_external` ones are the external momenta, the remaining ones are
    /// the loop momenta. Returns an error if the number of labels does not match the topology.
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

    /// Generate the topologies.
    pub fn generate(&self) -> TopologyContainer {
        let degrees = self.model.degrees_iter().collect_vec();
        // Let N_k be the number of nodes with degree k, then
        //      \sum_{k=3}^\infty (k-2) N_k = 2 L - 2 + E                                        (1)
        // where L is the number of loops and E is the number of external particles.
        // The full set of diagrams is then the sum of all node partitions {N_k}, such that (1) is satisfied.
        let node_partitions = find_partitions(
            self.model.degrees_iter().map(|d| d - 2),
            2 * self.n_loops + self.n_external - 2,
        )
        .into_iter()
        .filter(|partition| {
            self.selector.select_partition(
                partition
                    .iter()
                    .enumerate()
                    .map(|(i, count)| (degrees[i], *count))
                    .collect_vec(),
            )
        })
        .collect_vec();

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
                if let Some(ref labels) = self.momentum_labels {
                    workspace.set_momentum_labels(labels.clone());
                }
                return workspace.generate();
            })
            .collect_into_vec(&mut containers);
        return TopologyContainer::from(containers);
    }

    /// Generate and count the topologies without saving them
    pub fn count_topologies(&self) -> usize {
        let degrees = self.model.degrees_iter().collect_vec();
        let node_partitions = find_partitions(
            self.model.degrees_iter().map(|d| d - 2),
            2 * self.n_loops + self.n_external - 2,
        )
        .into_iter()
        .filter(|partition| {
            self.selector.select_partition(
                partition
                    .iter()
                    .enumerate()
                    .map(|(i, count)| (degrees[i], *count))
                    .collect_vec(),
            )
        })
        .collect_vec();
        let mut counts = Vec::new();
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
                return workspace.count();
            })
            .collect_into_vec(&mut counts);
        return counts.into_iter().sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use test_log::test;

    #[test]
    fn topology_generator_custom_function_test() {
        let model = TopologyModel::from(vec![3, 4]);
        let mut selector = TopologySelector::default();
        let filter = |topo: &Topology| -> bool {
            for edge in topo.edges.iter() {
                if edge.connected_nodes[0] == edge.connected_nodes[1] {
                    return false;
                }
            }
            return true;
        };
        selector.add_custom_function(Arc::new(filter));
        let generator = TopologyGenerator::new(2, 1, model, Some(selector));
        let topologies = generator.generate();
        assert_eq!(topologies.len(), 1);
    }

    #[test]
    fn topology_bridge_test() {
        let adjacency_matrix = SymmetricMatrix::from_vec(
            10,
            vec![
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 1, 2,
            ],
        );
        let degrees = vec![1, 1, 1, 1, 3, 3, 4, 4, 4, 4];
        let mut edges: Vec<Edge> = Vec::new();
        for i in 0..10 {
            for _ in 0..*adjacency_matrix.get(i, i) / 2 {
                edges.push(Edge::empty([i, i]))
            }
            for j in (i + 1)..10 {
                for _ in 0..*adjacency_matrix.get(i, j) {
                    edges.push(Edge::empty([i, j]))
                }
            }
        }
        let topo = Topology {
            n_external: 4,
            n_loops: 4,
            nodes: (0..10)
                .map(|i| Node::from_matrix(degrees[i], &adjacency_matrix, i))
                .collect_vec(),
            edges,
            node_symmetry: 1,
            edge_symmetry: 1,
            momentum_labels: vec!["p1", "p2", "p3", "p4", "l1", "l2", "l3", "l4"]
                .into_iter()
                .map(|x| x.to_string())
                .collect_vec(),
            bridges: vec![(5, 6), (8, 9)],
            node_classification: NodeClassification::empty(),
        };
        println!("{:#?}", topo);
        assert_eq!(
            topo.bridges().into_iter().collect::<HashSet<(usize, usize)>>(),
            HashSet::from([(5, 6), (8, 9)])
        );
    }

    #[test]
    fn topology_momentum_assignment_test() {
        let topo_ref = Topology {
            n_external: 4,
            n_loops: 2,
            nodes: vec![
                Node::new(1, vec![6]),
                Node::new(1, vec![4]),
                Node::new(1, vec![5]),
                Node::new(1, vec![7]),
                Node::new(3, vec![1, 5, 6]),
                Node::new(4, vec![2, 4, 6, 7]),
                Node::new(4, vec![0, 4, 5, 7]),
                Node::new(3, vec![3, 5, 6]),
            ],
            edges: vec![
                Edge::new([0, 6], Some(vec![1, 0, 0, 0, 0, 0])),
                Edge::new([1, 4], Some(vec![0, 1, 0, 0, 0, 0])),
                Edge::new([2, 5], Some(vec![0, 0, 1, 0, 0, 0])),
                Edge::new([3, 7], Some(vec![0, 0, 0, 1, 0, 0])),
                Edge::new([4, 5], Some(vec![1, 1, 0, 0, 1, 1])),
                Edge::new([4, 6], Some(vec![-1, 0, 0, 0, -1, -1])),
                Edge::new([5, 6], Some(vec![0, 0, 0, 0, 1, 0])),
                Edge::new([5, 7], Some(vec![1, 1, 1, 0, 0, 1])),
                Edge::new([6, 7], Some(vec![0, 0, 0, 0, 0, -1])),
            ],
            node_symmetry: 1,
            edge_symmetry: 4,
            momentum_labels: vec![
                String::from("p1"),
                String::from("p2"),
                String::from("p3"),
                String::from("p4"),
                String::from("l1"),
                String::from("l2"),
            ],
            bridges: vec![],
            node_classification: NodeClassification::empty(),
        };
        let mut topo = Topology {
            n_external: 4,
            n_loops: 2,
            nodes: vec![
                Node::new(1, vec![6]),
                Node::new(1, vec![4]),
                Node::new(1, vec![5]),
                Node::new(1, vec![7]),
                Node::new(3, vec![1, 5, 6]),
                Node::new(4, vec![2, 4, 6, 7]),
                Node::new(4, vec![0, 4, 5, 7]),
                Node::new(3, vec![3, 5, 6]),
            ],
            edges: vec![
                Edge::new([0, 6], None),
                Edge::new([1, 4], None),
                Edge::new([2, 5], None),
                Edge::new([3, 7], None),
                Edge::new([4, 5], None),
                Edge::new([4, 6], None),
                Edge::new([5, 6], None),
                Edge::new([5, 7], None),
                Edge::new([6, 7], None),
            ],
            node_symmetry: 1,
            edge_symmetry: 4,
            momentum_labels: vec![
                String::from("p1"),
                String::from("p2"),
                String::from("p3"),
                String::from("p4"),
                String::from("l1"),
                String::from("l2"),
            ],
            bridges: vec![],
            node_classification: NodeClassification::empty(),
        };
        topo.assign_momenta();
        assert_eq!(topo, topo_ref);
    }

    #[test]
    fn topology_momentum_assignment_test_2() {
        let topo_ref = Topology {
            n_external: 4,
            n_loops: 1,
            nodes: vec![
                Node::new(1, vec![4]),
                Node::new(1, vec![4]),
                Node::new(1, vec![6]),
                Node::new(1, vec![6]),
                Node::new(3, vec![0, 1, 5]),
                Node::new(3, vec![4, 6, 7]),
                Node::new(3, vec![2, 3, 5]),
                Node::new(3, vec![5, 7]),
            ],
            edges: vec![
                Edge::new([0, 4], Some(vec![1, 0, 0, 0, 0])),
                Edge::new([1, 4], Some(vec![0, 1, 0, 0, 0])),
                Edge::new([2, 6], Some(vec![0, 0, 1, 0, 0])),
                Edge::new([3, 6], Some(vec![0, 0, 0, 1, 0])),
                Edge::new([4, 5], Some(vec![1, 1, 0, 0, 0])),
                Edge::new([5, 6], Some(vec![1, 1, 0, 0, 0])),
                Edge::new([5, 7], Some(vec![0, 0, 0, 0, 0])),
                Edge::new([7, 7], Some(vec![0, 0, 0, 0, 1])),
            ],
            node_symmetry: 1,
            edge_symmetry: 2,
            momentum_labels: vec![
                String::from("p1"),
                String::from("p2"),
                String::from("p3"),
                String::from("p4"),
                String::from("l1"),
            ],
            bridges: vec![(5, 6), (5, 7), (4, 5)],
            node_classification: NodeClassification::empty(),
        };
        let mut topo = Topology {
            n_external: 4,
            n_loops: 1,
            nodes: vec![
                Node::new(1, vec![4]),
                Node::new(1, vec![4]),
                Node::new(1, vec![6]),
                Node::new(1, vec![6]),
                Node::new(3, vec![0, 1, 5]),
                Node::new(3, vec![4, 6, 7]),
                Node::new(3, vec![2, 3, 5]),
                Node::new(3, vec![5, 7]),
            ],
            edges: vec![
                Edge::new([0, 4], None),
                Edge::new([1, 4], None),
                Edge::new([2, 6], None),
                Edge::new([3, 6], None),
                Edge::new([4, 5], None),
                Edge::new([5, 6], None),
                Edge::new([5, 7], None),
                Edge::new([7, 7], None),
            ],
            node_symmetry: 1,
            edge_symmetry: 2,
            momentum_labels: vec![
                String::from("p1"),
                String::from("p2"),
                String::from("p3"),
                String::from("p4"),
                String::from("l1"),
            ],
            bridges: vec![],
            node_classification: NodeClassification::empty(),
        };
        topo.assign_momenta();
        assert_eq!(topo, topo_ref);
    }
}
