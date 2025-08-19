use crate::topology::filter::TopologySelector;
use crate::topology::{Topology, TopologyContainer};
use crate::topology::{
    components::{NodeClassification, TopologyNode},
    matrix::SymmetricMatrix,
};
use crate::util::generate_permutations;
use itertools::{FoldWhile, Itertools};
use std::cmp::{Ordering, min};

/// Workspace struct for the generation of all topologies of a given list of nodes.
#[derive(Debug)]
pub struct TopologyWorkspace {
    pub(crate) n_external: usize,
    pub(crate) n_loops: usize,
    /// List of nodes in ascending order by their degrees
    pub(crate) nodes: Vec<TopologyNode>,
    pub(crate) adjacency_matrix: SymmetricMatrix,
    /// List containing the current spanning forest of the nodes
    connection_forest: Vec<Option<usize>>,
    pub(crate) connection_components: usize,
    /// Total number of edges remaining to be added to the topology
    remaining_edges: usize,
    /// Current node classification
    pub(crate) node_classification: NodeClassification,
    topology_buffer: Option<TopologyContainer>,
    /// [TopologySelector] deciding whether a found topology is kept
    pub(crate) topology_selector: TopologySelector,
    /// List of names of the momenta
    pub(crate) momentum_labels: Vec<String>,
    /// Number of generated diagrams
    pub(crate) count: usize,
}

impl TopologyWorkspace {
    /// Create new workspace for generating the topologies of `n_external` external particles and number and
    /// degrees of nodes given by `node_degrees`.
    pub fn from_nodes(n_external: usize, n_loops: usize, node_degrees: &[usize]) -> Self {
        let node_degrees_sorted = node_degrees.iter().copied().sorted().collect_vec();
        let node_classification = NodeClassification::from_degrees(&node_degrees_sorted);
        let mut nodes: Vec<TopologyNode> = Vec::with_capacity(node_degrees.len());
        for degree in node_degrees_sorted {
            nodes.push(TopologyNode::empty(degree));
        }
        return Self {
            n_external,
            n_loops,
            nodes,
            connection_forest: vec![None; node_degrees.len()],
            connection_components: node_degrees.len(),
            adjacency_matrix: SymmetricMatrix::zero(node_degrees.len()),
            remaining_edges: node_degrees.iter().sum::<usize>() / 2,
            node_classification,
            topology_buffer: None,
            topology_selector: TopologySelector::default(),
            momentum_labels: vec![
                (1..=n_external).map(|i| format!("p{}", i)).collect_vec(),
                (1..=n_loops).map(|i| format!("l{}", i)).collect_vec(),
            ]
            .into_iter()
            .flatten()
            .collect_vec(),
            count: 0,
        };
    }

    /// Set the names of the momenta
    pub(crate) fn set_momentum_labels(&mut self, labels: Vec<String>) {
        self.momentum_labels = labels;
    }

    /// Return the root of the spanning tree to which `node` belongs. Flattens the tree in the process.
    fn find_root(&mut self, node: usize) -> usize {
        let mut current = node;
        while let Some(parent) = self.connection_forest[current] {
            if let Some(grandparent) = self.connection_forest[parent] {
                self.connection_forest[current] = Some(grandparent);
            }
            current = parent;
        }
        return current;
    }

    /// Get all nodes to which `node` has an edge, except itself if `node` has a self-loop.
    fn get_connections(&self, node: usize) -> Vec<usize> {
        return (0..self.nodes.len())
            .filter(|j| *j != node && *self.adjacency_matrix.get(node, *j) != 0)
            .collect();
    }

    /// Get all nodes to which `node` is connected, except itself if `node` has a self-loop.
    fn find_connected_nodes(&self, node: usize) -> Vec<usize> {
        let mut visited: Vec<usize> = vec![node];
        let mut to_visit: Vec<usize> = vec![node];
        let mut current;
        while !to_visit.is_empty() {
            current = to_visit.pop().unwrap();
            for node in self.get_connections(current) {
                if !visited.contains(&node) {
                    visited.push(node);
                    to_visit.push(node);
                }
            }
        }
        return visited;
    }

    /// Check whether the connection component to which `node` belongs has no remaining open connections and
    /// will thus remain disconnected.
    fn is_disconnected(&self, node: usize) -> bool {
        return self
            .find_connected_nodes(node)
            .into_iter()
            .map(|i| self.nodes[i].open_connections)
            .sum::<usize>()
            == 0;
    }

    /// Add a connection of `multiplicity` between `first_node` and `second_node`, updating the adjacency matrix,
    /// the spanning forest and the classification matrix in the process.
    fn add_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        if first_node != second_node {
            *self.adjacency_matrix.get_mut(first_node, second_node) += multiplicity;
            let first_root = self.find_root(first_node);
            let second_root = self.find_root(second_node);
            if first_root != second_root {
                // Nodes belonged to different connection components
                // -> merge both components, since they are now connected
                self.connection_forest[second_root] = Some(first_root);
                self.connection_components -= 1;
            }
        } else {
            *self.adjacency_matrix.get_mut(first_node, second_node) += 2 * multiplicity;
        }
        self.nodes[first_node].open_connections -= multiplicity;
        self.nodes[second_node].open_connections -= multiplicity;
        self.remaining_edges -= multiplicity;
        self.node_classification
            .add_connection(first_node, second_node, multiplicity);
    }

    /// Remove a connection of `multiplicity` between `first_node` and `second_node`, updating the adjacency
    /// matrix, the spanning forest and the classification matrix in the process.
    fn remove_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        if first_node != second_node {
            *self.adjacency_matrix.get_mut(first_node, second_node) -= multiplicity;
            if *self.adjacency_matrix.get(first_node, second_node) == 0 {
                // The nodes are not directly connected anymore - check if there is still an indirect connection
                // between them, otherwise split the connection component and (re-)build both spanning trees
                let first_component = self.find_connected_nodes(first_node);
                if !first_component.contains(&second_node) {
                    let second_component = self.find_connected_nodes(second_node);
                    self.connection_forest[first_node] = None;
                    self.connection_forest[second_node] = None;
                    for node in second_component.iter().skip(1) {
                        self.connection_forest[*node] = Some(second_node);
                    }
                    for node in first_component.iter().skip(1) {
                        self.connection_forest[*node] = Some(first_node);
                    }
                    self.connection_components += 1;
                }
            }
        } else {
            *self.adjacency_matrix.get_mut(first_node, second_node) -= 2 * multiplicity;
        }
        self.nodes[first_node].open_connections += multiplicity;
        self.nodes[second_node].open_connections += multiplicity;
        self.remaining_edges += multiplicity;
        self.node_classification
            .remove_connection(first_node, second_node, multiplicity);
    }

    /// Find the next class used as initial class for constructing new edges. This is always the first class
    /// with open connections, except if there is a class which already has connections, but is not saturated.
    /// Then this class is chosen. Since nodes with different connection structures are placed in different
    /// classes by the refinement procedure, only the first node of each class has to be considered.
    fn find_next_class(&self) -> Option<usize> {
        let mut next_class = None;
        for (class, boundary) in self
            .node_classification
            .boundaries
            .iter()
            .enumerate()
            .take(self.node_classification.n_classes())
        {
            match self.nodes[*boundary].open_connections {
                0 => (),
                open_connections if open_connections < self.nodes[*boundary].max_connections => return Some(class),
                _ if next_class.is_none() => next_class = Some(class),
                _ => (),
            }
        }
        return next_class;
    }

    /// Find the next class acting as target for the constructed edge. This is always the first node with
    /// open connections not excluded by `excluded_nodes`.
    fn find_next_target_class(&self, excluded_nodes: &[bool]) -> Option<usize> {
        for (index, node) in self.nodes.iter().enumerate() {
            if excluded_nodes[index] {
                continue;
            }
            match node.open_connections {
                0 => (),
                _ => {
                    return Some(self.node_classification.find_class(index));
                }
            }
        }
        return None;
    }

    /// Check whether a found graph is a representative of its orbit. This is decided by regarding the entries
    /// of the adjacency matrix as digits of a number, different graphs are then ordered by the size of this
    /// number. The graph is a representative if its coding is larger or equal to the codings of all other
    /// permutations of the adjacency matrix. Since the nodes are classified by their topological properties,
    /// only permutations within a class have to be considered.
    ///
    /// When the graph is the representative, the symmetry number by node permutations is returned.
    fn is_representative(&self) -> Option<usize> {
        return generate_permutations(&self.node_classification.get_class_sizes())
            .fold_while(Some(0usize), |acc, permutation| {
                match self.adjacency_matrix.cmp_permutation(&permutation) {
                    Ordering::Equal => FoldWhile::Continue(Some(acc.unwrap() + 1)),
                    Ordering::Greater => FoldWhile::Continue(Some(acc.unwrap())),
                    _ => FoldWhile::Done(None),
                }
            })
            .into_inner();
    }

    /// Find and connect the next class.
    fn connect_next_class(&mut self) {
        // First update the classification. If the classification from the current configuration is
        // inconsistent, return immediately.
        if let Some(next_classification) = self.node_classification.update_classification(&self.adjacency_matrix) {
            let previous_classification = self.node_classification.clone(); // Save current classification
            self.node_classification = next_classification;
            if let Some(class) = self.find_next_class() {
                // Find the next class to be connected
                // Start by connecting the first node of the class
                self.connect_node(class, self.node_classification.boundaries[class]);
            } else {
                // If there is no class remaining to be connected, the graph is fully constructed
                if self.connection_components == 1
                    && let Some(node_symmetry) = self.is_representative()
                {
                    // Only keep fully connected graphs which are representatives
                    let topology = Topology::from(self, node_symmetry);
                    if self.topology_selector.select(&topology) {
                        if self.topology_buffer.is_some() {
                            self.topology_buffer.as_mut().unwrap().push(topology);
                        }
                        self.count += 1;
                    }
                }
            }
            self.node_classification = previous_classification; // Restore previous classification for further connections
        }
    }

    /// Try to connect `node`, or another remaining node in the class otherwise.
    fn connect_node(&mut self, class: usize, node: usize) {
        if node >= self.node_classification.boundaries[class + 1] {
            // If the requested node is not part of the given class, start connecting the next class
            self.connect_next_class();
        }
        // Try connecting the remaining nodes in the class
        for class_node in node..self.node_classification.boundaries[class + 1] {
            // Graph cannot become fully connected by adding edges if this is not satisfied
            if self.connection_components - 1 <= self.remaining_edges {
                self.connect_leg(class, class_node, class, &vec![false; self.nodes.len()]);
                return;
            }
        }
    }

    /// Connect a leg of `node` to `target_class`, or to the next target class if not possible. All nodes
    /// given in `skip_nodes` are ignored.
    fn connect_leg(&mut self, class: usize, node: usize, target_class: usize, skip_nodes: &Vec<bool>) {
        let mut current_target_class = target_class;
        let mut current_skip_nodes: Vec<bool> = (*skip_nodes).clone();

        if self.nodes[node].open_connections == 0 {
            // Return immediately if the graph cannot become fully connected
            if self.remaining_edges < (self.connection_components - 1)
                || (self.connection_components > 1 && self.is_disconnected(node))
            {
                return;
            }
            // No remaining connections for `node` -> start connecting next node
            self.connect_node(class, node + 1);
        } else {
            let mut advance_class = false;
            for _ in 0..self.nodes.len() {
                if class != current_target_class || advance_class {
                    // Find next class to connect `node` to, if there is none left, the graph is fully constructed
                    if let Some(next_target_class) = self.find_next_target_class(&current_skip_nodes) {
                        current_target_class = next_target_class;
                    } else {
                        return;
                    }
                }
                advance_class = true;
                for target_node in self.node_classification.class_iter(current_target_class) {
                    if current_skip_nodes[target_node] {
                        continue;
                    } else {
                        // All possible connection configurations for `target_node` will be constructed by the
                        // current function - therefore all calls of `connect_leg` deeper in the recursion can
                        // ignore `target_node`
                        current_skip_nodes[target_node] = true;
                    }
                    if class == current_target_class && target_node < node {
                        continue;
                    } else if target_node == node {
                        // Construct self-loops
                        for multiplicity in if self.nodes[node].open_connections == self.nodes[node].max_connections
                            && self.nodes.len() > 1
                        {
                            // Node is completely disconnected from any other node
                            // -> at least one connection has to remain open in order to generate a connected graph
                            // Only exception is for vacuum diagrams containing only a single node
                            1..=min(
                                (self.nodes[node].max_connections - 1) / 2,
                                self.remaining_edges - (self.connection_components - 1),
                            )
                        } else {
                            1..=min(
                                self.nodes[node].open_connections / 2,
                                self.remaining_edges - (self.connection_components - 1),
                            )
                        }
                        .rev()
                        {
                            self.add_connection(node, node, multiplicity);
                            self.connect_leg(
                                class,
                                node,
                                self.node_classification.find_class(target_node + 1),
                                &current_skip_nodes,
                            );
                            self.remove_connection(node, node, multiplicity);
                            advance_class = false;
                        }
                    } else {
                        // Construct edges between `node` and `target_node`
                        for multiplicity in if self.nodes[node].open_connections == self.nodes[node].max_connections
                            && self.nodes[target_node].open_connections == self.nodes[node].max_connections
                            && self.nodes[node].max_connections == self.nodes[target_node].max_connections
                            && self.nodes.len() > 2
                        {
                            // Both nodes are isolated from the remaining graph and have the same number of legs
                            // -> at least one connection has to remain open in order to generate a connected graph
                            // Only exception is for vacuum diagrams with exactly two nodes
                            1..=(self.nodes[node].max_connections - 1)
                        } else {
                            1..=min(
                                self.nodes[node].open_connections,
                                self.nodes[target_node].open_connections,
                            )
                        }
                        .rev()
                        {
                            self.add_connection(node, target_node, multiplicity);
                            self.connect_leg(
                                class,
                                node,
                                self.node_classification.find_class(target_node + 1),
                                &current_skip_nodes,
                            );
                            self.remove_connection(node, target_node, multiplicity);
                            advance_class = false;
                        }
                    }
                }
                if skip_nodes.iter().all(|x| *x) {
                    // Early exit if all nodes are taken care of
                    break;
                }
            }
        }
    }

    /// Generate all topologies of the current workspace.
    pub fn generate(&mut self) -> TopologyContainer {
        self.topology_buffer = Some(TopologyContainer::new());
        self.connect_next_class();
        let container = std::mem::take(&mut self.topology_buffer).unwrap();
        return container;
    }

    /// Generate and count all topologies of the current workspace without saving them.
    pub fn count(&mut self) -> usize {
        self.connect_next_class();
        return self.count;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::topology::matrix::SymmetricMatrix;
    use crate::topology::{Edge, Node, Topology, TopologyContainer};
    use itertools::Itertools;
    use test_log::test;

    #[test]
    fn connection_test() {
        let nodes: Vec<usize> = vec![1, 1, 1, 1, 4, 4];
        let mut graph = TopologyWorkspace::from_nodes(4, 1, &nodes);
        assert_eq!(graph.connection_components, 6);
        graph.add_connection(0, 4, 1);
        graph.add_connection(1, 5, 1);
        graph.add_connection(4, 5, 1);
        graph.add_connection(2, 3, 1);
        assert_eq!(graph.connection_components, 2);
        graph.remove_connection(4, 5, 1);
        assert_eq!(graph.connection_components, 3);
    }

    #[test]
    fn representative_test() {
        let nodes: Vec<usize> = vec![1, 4, 4, 1];
        let mut workspace = TopologyWorkspace::from_nodes(2, 2, &nodes);
        let adjacency_data: Vec<usize> = vec![0, 0, 1, 0, 0, 0, 1, 2, 1, 2];
        workspace.adjacency_matrix = SymmetricMatrix::from_vec(4, adjacency_data);
        assert!(workspace.is_representative().is_some());
        let adjacency_data: Vec<usize> = vec![0, 0, 0, 1, 0, 1, 0, 2, 1, 2];
        workspace.adjacency_matrix = SymmetricMatrix::from_vec(4, adjacency_data);
        assert!(workspace.is_representative().is_none());
    }

    #[test]
    fn topology_workspace_generate_test_1_loop() {
        let nodes: Vec<usize> = vec![1, 3, 3, 1];
        let selector = TopologySelector::new();
        let mut workspace = TopologyWorkspace::from_nodes(2, 1, &nodes);
        workspace.topology_selector = selector;
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 2);
    }

    #[test]
    fn topology_workspace_generate_test_2_loop() {
        let nodes: Vec<usize> = vec![1, 4, 4, 1];
        let selector = TopologySelector::new();
        let mut workspace = TopologyWorkspace::from_nodes(2, 2, &nodes);
        workspace.topology_selector = selector;
        let topologies = workspace.generate();
        let topologies_ref = TopologyContainer {
            data: vec![
                Topology {
                    n_external: 2,
                    n_loops: 2,
                    nodes: vec![
                        Node::new(1, vec![2]),
                        Node::new(1, vec![3]),
                        Node::new(4, vec![0, 2, 3]),
                        Node::new(4, vec![1, 2, 3]),
                    ],
                    edges: vec![
                        Edge::new([0, 2], Some(vec![1, 0, 0, 0])),
                        Edge::new([1, 3], Some(vec![0, 1, 0, 0])),
                        Edge::new([2, 2], Some(vec![0, 0, 1, 0])),
                        Edge::new([2, 3], Some(vec![1, 0, 0, 0])),
                        Edge::new([3, 3], Some(vec![0, 0, 0, 1])),
                    ],
                    node_symmetry: 1,
                    edge_symmetry: 4,
                    momentum_labels: vec![
                        String::from("p1"),
                        String::from("p2"),
                        String::from("l1"),
                        String::from("l2"),
                    ],
                    bridges: vec![(2, 3)],
                    node_classification: NodeClassification {
                        boundaries: vec![0, 1, 2, 3, 4],
                        matrix: vec![vec![0, 0, 1, 0], vec![0, 0, 0, 1], vec![1, 0, 2, 1], vec![0, 1, 1, 2]],
                    },
                },
                Topology {
                    n_external: 2,
                    n_loops: 2,
                    nodes: vec![
                        Node::new(1, vec![2]),
                        Node::new(1, vec![2]),
                        Node::new(4, vec![0, 1, 3]),
                        Node::new(4, vec![2, 3]),
                    ],
                    edges: vec![
                        Edge::new([0, 2], Some(vec![1, 0, 0, 0])),
                        Edge::new([1, 2], Some(vec![0, 1, 0, 0])),
                        Edge::new([2, 3], Some(vec![0, 0, 0, 1])),
                        Edge::new([2, 3], Some(vec![0, 0, 0, -1])),
                        Edge::new([3, 3], Some(vec![0, 0, 1, 0])),
                    ],
                    node_symmetry: 1,
                    edge_symmetry: 4,
                    momentum_labels: vec![
                        String::from("p1"),
                        String::from("p2"),
                        String::from("l1"),
                        String::from("l2"),
                    ],
                    bridges: vec![],
                    node_classification: NodeClassification {
                        boundaries: vec![0, 1, 2, 3, 4],
                        matrix: vec![vec![0, 0, 1, 0], vec![0, 0, 1, 0], vec![1, 1, 0, 2], vec![0, 0, 2, 2]],
                    },
                },
                Topology {
                    n_external: 2,
                    n_loops: 2,
                    nodes: vec![
                        Node::new(1, vec![2]),
                        Node::new(1, vec![3]),
                        Node::new(4, vec![0, 3]),
                        Node::new(4, vec![1, 2]),
                    ],
                    edges: vec![
                        Edge::new([0, 2], Some(vec![1, 0, 0, 0])),
                        Edge::new([1, 3], Some(vec![0, 1, 0, 0])),
                        Edge::new([2, 3], Some(vec![1, 0, 1, 0])),
                        Edge::new([2, 3], Some(vec![0, 0, -1, 1])),
                        Edge::new([2, 3], Some(vec![0, 0, 0, -1])),
                    ],
                    node_symmetry: 1,
                    edge_symmetry: 6,
                    momentum_labels: vec![
                        String::from("p1"),
                        String::from("p2"),
                        String::from("l1"),
                        String::from("l2"),
                    ],
                    bridges: vec![],
                    node_classification: NodeClassification {
                        boundaries: vec![0, 1, 2, 3, 4],
                        matrix: vec![vec![0, 0, 1, 0], vec![0, 0, 0, 1], vec![1, 0, 0, 3], vec![0, 1, 3, 0]],
                    },
                },
            ],
        };
        assert_eq!(topologies, topologies_ref);
    }

    #[test]
    fn topology_workspace_generate_test_3point_4_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 4]].into_iter().flatten().collect_vec();
        let selector = TopologySelector::new();
        let mut workspace = TopologyWorkspace::from_nodes(2, 2, &nodes);
        workspace.topology_selector = selector;
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 10);
    }

    #[test]
    fn topology_workspace_generate_test_3point_6_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 6]].into_iter().flatten().collect_vec();
        let selector = TopologySelector::new();
        let mut workspace = TopologyWorkspace::from_nodes(2, 3, &nodes);
        workspace.topology_selector = selector;
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 66);
    }

    #[test]
    fn topology_workspace_generate_test_3point_8_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 8]].into_iter().flatten().collect_vec();
        let selector = TopologySelector::new();
        let mut workspace = TopologyWorkspace::from_nodes(2, 4, &nodes);
        workspace.topology_selector = selector;
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 511);
    }

    #[test]
    fn topology_workspace_generate_test_3point_10_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 10]]
            .into_iter()
            .flatten()
            .collect_vec();
        let selector = TopologySelector::new();
        let mut workspace = TopologyWorkspace::from_nodes(2, 6, &nodes);
        workspace.topology_selector = selector;
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 4536);
    }

    #[test]
    fn topology_workspace_generate_test_3point_12_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 12]]
            .into_iter()
            .flatten()
            .collect_vec();
        let selector = TopologySelector::new();
        let mut workspace = TopologyWorkspace::from_nodes(2, 6, &nodes);
        workspace.topology_selector = selector;
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 45519);
    }
}
