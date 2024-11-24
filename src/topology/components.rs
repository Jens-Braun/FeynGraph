use std::cmp::{min, Ordering};
use crate::topology::matrix::SymmetricMatrix;
use itertools::{izip, Itertools};
use itertools::FoldWhile;
use crate::topology::{Topology, TopologyContainer};
use rayon::prelude::*;
use crate::util::generate_permutations;

#[derive(Debug, Copy, Clone)]
struct TopologyNode {
    pub max_connections: usize,
    pub open_connections: usize
}

impl TopologyNode {
    pub fn empty(max_connections: usize) -> Self {
        return Self {
            max_connections,
            open_connections: max_connections
        }
    }
}

#[derive(Debug, Clone)]
struct NodeClassification {
    pub boundaries: Vec<usize>,
    pub matrix: Vec<Vec<usize>>
}

impl NodeClassification {
    pub fn from_degrees(node_degrees: &Vec<usize>) -> Self {
        let mut boundaries: Vec<usize> = Vec::with_capacity(node_degrees.len());
        let mut previous = node_degrees[0];
        for (node_index, node_degree) in node_degrees.iter().enumerate() {
            if *node_degree == 1 || *node_degree != previous {
                boundaries.push(node_index);
                previous = *node_degree;
            }
        }
        boundaries.push(node_degrees.len());
        let n_boundaries = boundaries.len()-1;
        return Self {
            boundaries,
            matrix: vec![vec![0; n_boundaries]; node_degrees.len()]
        }
    }

    pub fn update_classification_matrix(&mut self, adjacency_matrix: &SymmetricMatrix<usize>) {
        let n_classes = self.n_classes();
        if n_classes != self.matrix[0].len() {
            for row in &mut self.matrix {
                *row = vec![0; n_classes];
            }
        }
        for (node_index, node_vector) in self.matrix.iter_mut().enumerate() {
            for (class, (start, end)) in self.boundaries.iter().tuple_windows().enumerate() {
                node_vector[class] = (*start..*end).map(|i| adjacency_matrix.get(node_index, i)).sum();
            }
        }
    }

    pub fn add_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class= self.find_class(first_node);
        let second_class= self.find_class(second_node);
        self.matrix[first_node][second_class] += multiplicity;
        self.matrix[second_node][first_class] += multiplicity;
    }

    pub fn remove_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class= self.find_class(first_node);
        let second_class= self.find_class(second_node);
        self.matrix[first_node][second_class] -= multiplicity;
        self.matrix[second_node][first_class] -= multiplicity;
    }

    pub fn find_class(&self, node: usize) -> usize {
        for (class, boundary) in self.boundaries.iter().enumerate() {
            if *boundary > node {
                return class-1;
            }
        }
        return self.boundaries.len()-1;
    }

    pub fn n_classes(&self) -> usize {
        return self.boundaries.len()-1;
    }

    pub fn get_partition_sizes(&self) -> Vec<usize> {
        return self.boundaries.iter().tuple_windows().map(|(start, end)| *end - *start).collect_vec();
    }

    pub fn class_iter(&self, class: usize) -> impl Iterator<Item=usize> {
        return self.boundaries[class]..self.boundaries[class+1];
    }

    fn compare_node_classification(&self, first_node: usize, second_node: usize) -> Ordering {
        return match self.find_class(first_node).cmp(&self.find_class(second_node)) {
            Ordering::Equal => {
                for (x, y) in izip!(&self.matrix[first_node],
                                                   &self.matrix[second_node]) {
                    match x.cmp(y) {
                        Ordering::Equal => (),
                        ord => return ord.reverse()
                    }
                }
                return Ordering::Equal;
            },
            Ordering::Less => Ordering::Equal,
            Ordering::Greater => Ordering::Greater,
        }
    }

    fn refine_classification(&self, adjacency_matrix: &SymmetricMatrix<usize>) -> Option<Self> {
        let mut classification = (*self).clone();
        let mut new_boundaries: Vec<(usize, usize)> = Vec::new();
        let mut rerun = true;
        while rerun {
            rerun = false;
            for class in 0..classification.n_classes() {
                if classification.boundaries[class+1] - classification.boundaries[class] == 1 {
                    continue;
                }
                for node in classification.boundaries[class]..(classification.boundaries[class+1]-1) {
                    match classification.compare_node_classification(node, node+1) {
                        Ordering::Equal => (),
                        Ordering::Less => {
                            new_boundaries.push((class+1, node+1));
                            rerun = true;
                        },
                        Ordering::Greater => return None
                    }
                }
            }
            for (i, (class, node)) in new_boundaries.drain(..).enumerate() {
                classification.boundaries.insert(class+i, node);
            }
            classification.update_classification_matrix(adjacency_matrix);
        }
        return Some(classification);
    }
}

#[derive(Debug)]
pub struct TopologyWorkspace{
    nodes: Vec<TopologyNode>,
    pub adjacency_matrix: SymmetricMatrix<usize>,
    connection_tree: Vec<Option<usize>>,
    pub connection_components: usize,
    remaining_edges: usize,
    node_classification: NodeClassification,
    topology_buffer: Option<TopologyContainer>,
}

impl TopologyWorkspace {
    pub fn from_nodes(node_degrees: &Vec<usize>) -> Self {
        let node_degrees_sorted = node_degrees.clone().into_iter().sorted().collect_vec();
        let node_classification = NodeClassification::from_degrees(&node_degrees_sorted);
        let mut nodes: Vec<TopologyNode> = Vec::with_capacity(node_degrees.len());
        for degree in node_degrees_sorted.iter() {
            nodes.push(TopologyNode::empty(*degree));
        }
        return Self {
            nodes,
            connection_tree: vec![None; node_degrees.len()],
            connection_components: node_degrees.len(),
            adjacency_matrix: SymmetricMatrix::zero(node_degrees.len()),
            remaining_edges: node_degrees.iter().sum::<usize>()/2,
            node_classification,
            topology_buffer: None,
        }
    }

    fn find_root(&mut self, node: usize) -> usize {
        let mut current = node;
        loop {
            match self.connection_tree[current] {
                Some(parent) =>  {
                    if let Some(grandparent) = self.connection_tree[parent] {
                        self.connection_tree[current] = Some(grandparent);
                    }
                    current = parent;
                },
                None => break
            }
        }
        return current;
    }

    fn get_connections(&self, node: usize) -> Vec<usize> {
        return (0..self.nodes.len()).filter_map(
            |j| {
                if j != node && *self.adjacency_matrix.get(node, j) != 0 {
                    Some(j)
                } else { None }
            }
        ).collect();
    }

    fn find_connected_nodes(&self, node: usize) -> Vec<usize> {
        let mut visited: Vec<usize> = vec![node];
        let mut to_visit: Vec<usize> = vec![node];
        let mut current;
        while to_visit.len() > 0 {
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

    fn is_disconnected(&self, node: usize) -> bool {
        return self.find_connected_nodes(node)
            .into_iter()
            .map(|i| self.nodes[i].open_connections)
            .sum::<usize>() == 0;
    }

    fn add_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        if first_node != second_node {
            *self.adjacency_matrix.get_mut(first_node, second_node) += multiplicity;
            let first_root = self.find_root(first_node);
            let second_root = self.find_root(second_node);
            if first_root != second_root {
                self.connection_tree[second_root] = Some(first_root);
                self.connection_components -= 1;
            }
        } else {
            *self.adjacency_matrix.get_mut(first_node, second_node) += 2*multiplicity;
        }
        self.nodes[first_node].open_connections -= multiplicity;
        self.nodes[second_node].open_connections -= multiplicity;
        self.remaining_edges -= multiplicity;
        self.node_classification.add_connection(first_node, second_node, multiplicity);
    }

    fn remove_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        if first_node != second_node {
            *self.adjacency_matrix.get_mut(first_node, second_node) -= multiplicity;
            if *self.adjacency_matrix.get(first_node, second_node) == 0 {
                let first_component = self.find_connected_nodes(first_node);
                if !first_component.contains(&second_node) {
                    let second_component = self.find_connected_nodes(second_node);
                    self.connection_tree[first_node] = None;
                    self.connection_tree[second_node] = None;
                    for node in second_component.iter().skip(1) {
                        self.connection_tree[*node] = Some(second_node);
                    }
                    for node in first_component.iter().skip(1) {
                        self.connection_tree[*node] = Some(first_node);
                    }
                    self.connection_components += 1;
                }
            }
        } else {
            *self.adjacency_matrix.get_mut(first_node, second_node) -= 2*multiplicity;
        }
        self.nodes[first_node].open_connections += multiplicity;
        self.nodes[second_node].open_connections += multiplicity;
        self.remaining_edges += multiplicity;
        self.node_classification.remove_connection(first_node, second_node, multiplicity);
    }

    fn find_next_class(&self) -> Option<usize> {
        let mut next_class = None;
        for (class, boundary) in self.node_classification.boundaries.iter().enumerate()
            .take(self.node_classification.n_classes()) {
            match self.nodes[*boundary].open_connections {
                0 => (),
                open_connections if open_connections < self.nodes[*boundary].max_connections => return Some(class),
                _ if next_class.is_none() => next_class = Some(class),
                _ => (),
            }
        }
        return next_class;
    }

    fn find_next_target_class(&self, excluded_nodes: &Vec<bool>) -> Option<usize> {
        for node_index in 0..self.nodes.len() {
            if excluded_nodes[node_index] {
                continue;
            }
            match self.nodes[node_index].open_connections {
                0 => (),
                _ => { return Some(self.node_classification.find_class(node_index)); }
            }
        }
        return None;
    }

    fn is_representative(&self) -> Option<usize> {
        return generate_permutations(&self.node_classification.get_partition_sizes())
            // .par_bridge()
            // .map(|permutation| {
            //     match self.adjacency_matrix.cmp_permutation(&permutation) {
            //         Ordering::Equal => Some(1),
            //         Ordering::Greater => Some(0),
            //         _ => None
            //     }
            // })
            // .try_reduce(|| 0_usize, usize::checked_add);
            .fold_while(Some(0usize), |acc, permutation| {
                match self.adjacency_matrix.cmp_permutation(&permutation) {
                    Ordering::Equal => FoldWhile::Continue(Some(acc.unwrap() + 1)),
                    Ordering::Greater => FoldWhile::Continue(Some(acc.unwrap())),
                    _ => FoldWhile::Done(None)
                }
            }).into_inner();
    }

    fn connect_next_class(&mut self) {
        if let Some(next_classification) = self.node_classification.refine_classification(&self.adjacency_matrix) {
            let previous_classification = self.node_classification.clone();
            self.node_classification = next_classification;
            if let Some(class) = self.find_next_class() {
                self.connect_node(class, self.node_classification.boundaries[class]);
            } else {
                if self.connection_components == 1 {
                    if let Some(node_symmetry) = self.is_representative() {
                        let topology = Topology::from(&self, node_symmetry);
                        self.topology_buffer.as_mut().unwrap().push(topology);
                    }
                }
            }
            self.node_classification = previous_classification;
        }
    }

    fn connect_node(&mut self, class: usize, node: usize) {
        if node >= self.node_classification.boundaries[class+1] {
            self.connect_next_class();
        }
        for class_node in node..self.node_classification.boundaries[class+1] {
            if self.connection_components - 1 <= self.remaining_edges {
                self.connect_leg(class, class_node, class, &vec![false; self.nodes.len()]);
                return;
            } 
        }
    }

    fn connect_leg(&mut self, class: usize, node: usize, target_class: usize, skip_nodes: &Vec<bool>) {
        let mut current_target_class = target_class;
        let mut current_skip_nodes: Vec<bool> = (*skip_nodes).clone();

        if self.nodes[node].open_connections == 0 {
            if self.remaining_edges < (self.connection_components-1) ||
                (self.connection_components > 1 && self.is_disconnected(node)) { return; }
            self.connect_node(class, node+1);
        } else {
            let mut advance_class = false;
            for _ in 0..self.nodes.len() {
                if class != current_target_class || advance_class {
                    if let Some(next_target_class) = self.find_next_target_class(&current_skip_nodes) {
                        current_target_class = next_target_class;
                    } else { return; }
                }
                advance_class = true;
                for target_node in self.node_classification.class_iter(current_target_class) {
                    if current_skip_nodes[target_node] {
                        continue;
                    } else {
                        current_skip_nodes[target_node] = true;
                    }
                    if class == current_target_class && target_node < node {
                        continue;
                    } else if target_node == node {
                        for multiplicity in
                            if self.nodes[node].open_connections == self.nodes[node].max_connections {
                                // Node is completely disconnected from any other node
                                // -> at least one connection has to remain open in order to generate a connected graph
                                1..=min(
                                    (self.nodes[node].max_connections-1)/2,
                                    self.remaining_edges-(self.connection_components-1)
                                )
                            } else {
                                1..=min(
                                    self.nodes[node].open_connections/2,
                                    self.remaining_edges-(self.connection_components-1)
                                )
                            }.rev() {
                            self.add_connection(node, node, multiplicity);
                            self.connect_leg(class, node, self.node_classification.find_class(target_node + 1), &current_skip_nodes);
                            self.remove_connection(node, node, multiplicity);
                            advance_class = false;
                        }
                    } else {
                        for multiplicity in
                            if self.nodes[node].open_connections == self.nodes[node].max_connections
                            && self.nodes[target_node].open_connections == self.nodes[node].max_connections
                            && self.nodes[node].max_connections == self.nodes[target_node].max_connections {
                                // Both nodes are isolated from the remaining graph and have the same number of legs
                                // -> at least one connection has to remain open in order to generate a connected graph
                                1..=(self.nodes[node].max_connections - 1)
                            } else {
                                1..=min(self.nodes[node].open_connections, self.nodes[target_node].open_connections)
                            }.rev() {
                            self.add_connection(node, target_node, multiplicity);
                            self.connect_leg(class, node, self.node_classification.find_class(target_node + 1), &current_skip_nodes);
                            self.remove_connection(node, target_node, multiplicity);
                            advance_class = false;
                        }
                    }
                }
                if skip_nodes.iter().all(|x| *x) {
                    break;
                }
            }
        }
    }

    pub fn generate(&mut self) -> TopologyContainer {
        self.topology_buffer = Some(TopologyContainer::new());
        self.connect_next_class();
        let container = std::mem::take(&mut self.topology_buffer).unwrap();
        return container;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn connection_test () {
        let nodes: Vec<usize> = vec![1, 1, 1, 1, 4, 4];
        let mut graph = TopologyWorkspace::from_nodes(&nodes);
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
    fn representative_test () {
        let nodes: Vec<usize> = vec![1, 4, 4, 1];
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
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
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 2);
    }

    #[test]
    fn topology_workspace_generate_test_2_loop() {
        let nodes: Vec<usize> = vec![1, 4, 4, 1];
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
        let topologies = workspace.generate();
        let topologies_ref = TopologyContainer {
            data: vec![
                Topology {
                    adjacency_matrix: SymmetricMatrix::from_vec(4, vec![0, 0, 1, 0, 0, 0, 1, 2, 1, 2]),
                    node_symmetry: 1,
                    edge_symmetry: 4
                },
                Topology {
                    adjacency_matrix: SymmetricMatrix::from_vec(4, vec![0, 0, 1, 0, 0, 1, 0, 0, 2, 2]),
                    node_symmetry: 1,
                    edge_symmetry: 4
                },
                Topology {
                    adjacency_matrix: SymmetricMatrix::from_vec(4, vec![0, 0, 1, 0, 0, 0, 1, 0, 3, 0]),
                    node_symmetry: 1,
                    edge_symmetry: 6
                }
            ]
        };
        assert_eq!(topologies, topologies_ref);
    }

    #[test]
    fn topology_workspace_generate_test_3point_4_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 4]].into_iter().flatten().collect_vec();
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 10);
    }

    #[test]
    fn topology_workspace_generate_test_3point_6_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 6]].into_iter().flatten().collect_vec();
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 66);
    }

    #[test]
    fn topology_workspace_generate_test_3point_8_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 8]].into_iter().flatten().collect_vec();
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 511);
    }

    #[test]
    fn topology_workspace_generate_test_3point_10_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 10]].into_iter().flatten().collect_vec();
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 4536);
    }

    #[test]
    fn topology_workspace_generate_test_3point_12_vertices() {
        let nodes: Vec<usize> = [vec![1_usize; 2], vec![3_usize; 12]].into_iter().flatten().collect_vec();
        let mut workspace = TopologyWorkspace::from_nodes(&nodes);
        let topologies = workspace.generate();
        assert_eq!(topologies.inner_ref().len(), 45519);
    }
}