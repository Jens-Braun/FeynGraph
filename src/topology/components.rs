use std::cmp::Ordering;
use crate::topology::matrix::SymmetricMatrix;
use itertools::{izip, Itertools};


#[derive(Debug, Copy, Clone)]
pub(crate) struct TopologyNode {
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
pub(crate) struct NodeClassification {
    pub boundaries: Vec<usize>,
    pub matrix: Vec<Vec<usize>>
}

impl NodeClassification {
    pub(crate) fn from_degrees(node_degrees: &Vec<usize>) -> Self {
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

    fn update_classification_matrix(&mut self, adjacency_matrix: &SymmetricMatrix<usize>) {
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

    pub(crate) fn add_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class= self.find_class(first_node);
        let second_class= self.find_class(second_node);
        self.matrix[first_node][second_class] += multiplicity;
        self.matrix[second_node][first_class] += multiplicity;
    }

    pub(crate) fn remove_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class= self.find_class(first_node);
        let second_class= self.find_class(second_node);
        self.matrix[first_node][second_class] -= multiplicity;
        self.matrix[second_node][first_class] -= multiplicity;
    }

    pub(crate) fn find_class(&self, node: usize) -> usize {
        for (class, boundary) in self.boundaries.iter().enumerate() {
            if *boundary > node {
                return class-1;
            }
        }
        return self.boundaries.len()-1;
    }

    pub(crate) fn n_classes(&self) -> usize {
        return self.boundaries.len()-1;
    }

    pub(crate) fn get_partition_sizes(&self) -> Vec<usize> {
        return self.boundaries.iter().tuple_windows().map(|(start, end)| *end - *start).collect_vec();
    }

    pub(crate) fn class_iter(&self, class: usize) -> impl Iterator<Item=usize> {
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

    pub(crate) fn refine_classification(&self, adjacency_matrix: &SymmetricMatrix<usize>) -> Option<Self> {
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