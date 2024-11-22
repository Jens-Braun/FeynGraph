use std::cmp::Ordering;
use std::sync::atomic::AtomicBool;
use crate::topology::matrix::SymmetricMatrix;
use itertools::{izip, Itertools};
use crate::topology::Topology;
use rayon::prelude::*;
use crate::util::generate_permutations;

#[derive(Debug, Copy, Clone)]
struct TopologyNode {
    pub max_connections: usize,
    pub open_connections: usize,
    pub current_class: usize,
}

impl TopologyNode {
    pub fn empty(max_connections: usize, current_class: usize) -> Self {
        return Self {
            max_connections,
            open_connections: max_connections,
            current_class
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
        let n_boundaries = boundaries.len();
        return Self {
            boundaries,
            matrix: vec![vec![0; n_boundaries]; node_degrees.len()]
        }
    }

    pub fn update_classification_matrix(&mut self, adjacency_matrix: &SymmetricMatrix<usize>) {
        for (node_index, node_vector) in self.matrix.iter_mut().enumerate() {
            for (class, (start, end)) in self.boundaries.iter().tuple_windows().enumerate() {
                node_vector[class] = (*start..*end).map(|i| adjacency_matrix.get(node_index, i)).sum();
            }
        }
    }

    pub fn add_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class= self.find_class(first_node);
        let second_class= self.find_class(first_node);
        self.matrix[first_node][second_class] += multiplicity;
        self.matrix[second_node][first_class] += multiplicity;
    }

    pub fn remove_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class= self.find_class(first_node);
        let second_class= self.find_class(first_node);
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

    fn compare_node_classification(&self, first_node: usize, second_node: usize) -> Ordering {
        return match self.find_class(first_node).cmp(&self.find_class(second_node)) {
            Ordering::Equal => {
                for (x, y) in izip!(&self.matrix[first_node],
                                                   &self.matrix[second_node]) {
                    match x.cmp(y) {
                        Ordering::Equal => (),
                        ord => return ord
                    }
                }
                return Ordering::Equal;
            },
            ord => ord
        }
    }

    fn check_ordering(&self) -> bool {
        todo!();
    }

    fn refine_classification(&self, adjacency_matrix: &SymmetricMatrix<usize>) -> Option<Self> {
        let mut classification = (*self).clone();
        let mut rerun = true;
        while rerun {
            rerun = false;
            for class in 1..classification.boundaries.len() {
                if classification.boundaries[class] - classification.boundaries[class-1] == 1 {
                    continue;
                }
                for node in classification.boundaries[class-1]..classification.boundaries[class] {
                    match classification.compare_node_classification(node, node-1) {
                        Ordering::Equal => (),
                        Ordering::Less => {
                            classification.boundaries.insert(class+1, node);
                            classification.update_classification_matrix(adjacency_matrix);
                            rerun = true;
                        },
                        Ordering::Greater => return None
                    }
                }
            }
        }
        return Some(classification);
    }
}

pub struct TopologyWorkspace{
    nodes: Vec<TopologyNode>,
    pub adjacency_matrix: SymmetricMatrix<usize>,
    connection_tree: Vec<Option<usize>>,
    pub connection_components: usize,
    remaining_edges: usize,
    node_classification: NodeClassification,
    topology_list: Vec<Topology>
}

impl TopologyWorkspace {
    pub fn from_nodes(node_degrees: &Vec<usize>) -> Self {
        let node_degrees_sorted = node_degrees.clone().into_iter().sorted().collect_vec();
        let node_classification = NodeClassification::from_degrees(&node_degrees_sorted);
        let mut nodes: Vec<TopologyNode> = Vec::with_capacity(node_degrees.len());
        for (i, degree) in node_degrees_sorted.iter().enumerate() {
            nodes.push(TopologyNode::empty(*degree, node_classification.find_class(i)));
        }
        return Self {
            nodes,
            connection_tree: vec![None; node_degrees.len()],
            connection_components: node_degrees.len(),
            adjacency_matrix: SymmetricMatrix::zero(node_degrees.len()),
            remaining_edges: node_degrees.iter().sum(),
            node_classification,
            topology_list: Vec::new(),
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
        let mut visited: Vec<usize> = Vec::new();
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
        self.node_classification.add_connection(first_node, second_node, 1);
    }

    fn remove_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        if first_node != second_node {
            *self.adjacency_matrix.get_mut(first_node, second_node) -= multiplicity;
            if *self.adjacency_matrix.get(first_node, second_node) == 0 {
                let fist_component = self.find_connected_nodes(first_node);
                if !fist_component.contains(&second_node) {
                    let second_component = self.find_connected_nodes(second_node);
                    self.connection_tree[second_node] = None;
                    for node in second_component.iter().skip(1) {
                        self.connection_tree[*node] = Some(second_node);
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
        self.node_classification.remove_connection(first_node, second_node, 1);
    }

    fn find_next_class(&self) -> Option<usize> {
        for (class, boundary) in self.node_classification.boundaries.iter().enumerate() {
            match self.nodes[*boundary].open_connections {
                0 => (),
                _ => return Some(class)
            }
        }
        return None;
    }

    fn find_next_target_class(&self) -> Option<usize> {
        for node_index in 0..self.nodes.len() {
            match self.nodes[node_index].open_connections {
                0 => (),
                _ => return Some(self.node_classification.find_class(node_index))
            }
        }
        return None;
    }

    fn is_representative(&self) -> Option<usize> {
        let early_exit = AtomicBool::new(false);
        let count = generate_permutations(&self.node_classification.get_partition_sizes())
            .par_bridge()
            .take_any_while(|permutation| {
                match self.adjacency_matrix.cmp_permutation(permutation) {
                    Ordering::Equal => true,
                    _ => {
                        early_exit.store(true, std::sync::atomic::Ordering::Relaxed);
                        false
                    }
                }
            }).count();
        return if early_exit.into_inner() {
            None
        } else {
            Some(count)
        }
    }

    fn connect_next_class(&mut self) {
        if let Some(next_classification) = self.node_classification.refine_classification(&self.adjacency_matrix) {
            self.node_classification = next_classification;
            if let Some(class) = self.find_next_class() {
                self.connect_node(class, self.node_classification.boundaries[class]);
            } else {
                if self.connection_components == 0 {
                    if let Some(node_symmetry) = self.is_representative() {
                        self.topology_list.push(Topology::from(&self, node_symmetry));
                    }
                }
            }
        }
    }

    fn connect_node(&mut self, class: usize, node: usize) {
        if class < self.node_classification.n_classes() && node >= self.node_classification.boundaries[class+1] {
            self.connect_next_class();
        }
        for class_node in
            if class < self.node_classification.n_classes() {
                node..self.node_classification.boundaries[class+1]
            } else {
                node..self.nodes.len()
            } {
            self.connect_leg(class, class_node, class, class_node);
        }
    }

    fn connect_leg(&mut self, class: usize, node: usize, target_class: usize, target_node: usize) {
        if self.nodes[node].open_connections == 0 {
            if self.remaining_edges < self.connection_components { return; }
            self.connect_node(class, node+1);
        } else {
            if class != target_class {
                if let Some(next_target_class) = self.find_next_target_class() {
                    if !self.node_classification.check_ordering() {
                        return;
                    }
                } else { return; }
            }
        }
    }

    pub fn generate(&mut self) {
        self.connect_next_class();
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
}