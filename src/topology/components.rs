use crate::topology::matrix::SymmetricMatrix;
use itertools::{Itertools, izip};
use std::cmp::Ordering;

/// Node struct used internally by [TopologyWorkspace] during the generation of topologies
#[derive(Debug, Copy, Clone)]
pub(crate) struct TopologyNode {
    pub max_connections: usize,
    pub open_connections: usize,
}

impl TopologyNode {
    /// Create isolated node with degree `max_connections`
    pub fn empty(max_connections: usize) -> Self {
        return Self {
            max_connections,
            open_connections: max_connections,
        };
    }
}

/// Classification struct used by [TopologyWorkspace] during the generation of topologies to keep and refine the
/// current topological classification of the nodes. The classification only takes into account the topological
/// properties, i.e. the degree of the nodes, their current number of connections and in which other classes
/// these connections end.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NodeClassification {
    /// Boundaries of the node classifications. The $n$-th class has boundaries `boundaries[$n$]` and
    /// `boundaries[$n+1$]`, the last entry of `boundaries` is always the number of nodes plus one.
    pub boundaries: Vec<usize>,
    /// $N_\mathrm{nodes} \times N_\mathrm{classes}$ classification matrix. The entry $A_{ij}$ contains the number
    /// of edges connecting the $i$-th node with the $j$-th class.
    pub matrix: Vec<Vec<usize>>,
}

impl NodeClassification {
    /// Create an empty node classification
    pub(crate) fn empty() -> Self {
        return Self {
            boundaries: Vec::new(),
            matrix: Vec::new(),
        };
    }

    /// Create a new classification of a list of node degrees. The degrees are assumed to be sorted in ascending
    /// order. The initial classification is only based on the degrees, where nodes with identical degrees are
    /// placed in the same class. Nodes with degree $1$ are assumed to be external and are therefore placed
    /// in different classes.
    pub(crate) fn from_degrees(node_degrees: &[usize]) -> Self {
        let mut boundaries: Vec<usize> = Vec::with_capacity(node_degrees.len());
        let mut previous = node_degrees[0];
        for (node_index, node_degree) in node_degrees.iter().enumerate() {
            if *node_degree == 1 || node_index == 0 || *node_degree != previous {
                boundaries.push(node_index);
                previous = *node_degree;
            }
        }
        boundaries.push(node_degrees.len());
        let n_boundaries = boundaries.len() - 1;
        return Self {
            boundaries,
            matrix: vec![vec![0; n_boundaries]; node_degrees.len()],
        };
    }

    /// Construct the classification matrix for the current boundaries from the adjacencies given in
    /// `adjacency_matrix`.
    fn update_classification_matrix(&mut self, adjacency_matrix: &SymmetricMatrix) {
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

    /// Add a connection of `multiplicity` between `first_node` and `second_node` without updating the
    /// classification.
    pub(crate) fn add_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class = self.find_class(first_node);
        let second_class = self.find_class(second_node);
        self.matrix[first_node][second_class] += multiplicity;
        self.matrix[second_node][first_class] += multiplicity;
    }

    /// Remove a connection of `multiplicity` between `first_node` and `second_node` without updating the
    /// classification.
    pub(crate) fn remove_connection(&mut self, first_node: usize, second_node: usize, multiplicity: usize) {
        let first_class = self.find_class(first_node);
        let second_class = self.find_class(second_node);
        self.matrix[first_node][second_class] -= multiplicity;
        self.matrix[second_node][first_class] -= multiplicity;
    }

    /// Find the class to which `node` belongs.
    pub(crate) fn find_class(&self, node: usize) -> usize {
        for (class, boundary) in self.boundaries.iter().enumerate() {
            if *boundary > node {
                return class - 1;
            }
        }
        return self.boundaries.len() - 1;
    }

    pub(crate) fn n_classes(&self) -> usize {
        return self.boundaries.len() - 1;
    }

    /// Return a list of the sizes of the classes.
    pub(crate) fn get_class_sizes(&self) -> Vec<usize> {
        return self
            .boundaries
            .iter()
            .tuple_windows()
            .map(|(start, end)| *end - *start)
            .collect_vec();
    }

    /// Return an iterator over the nodes in `class`.
    pub(crate) fn class_iter(&self, class: usize) -> impl Iterator<Item = usize> + use<> {
        return self.boundaries[class]..self.boundaries[class + 1];
    }

    /// Compare $N_1$ (`first_node`) and $N_2$ (`second_node`) within the current classification. They are compared
    /// according to the classes $C_1$ and $C_2$ they belong to, and the classification matrix. If $C_1 > C_2$, then
    /// also $N_1 > N_2$. If $C_1 < C_2$, they are correctly ordered, which is represented by `Equal`. If they are
    /// in the same class, the ordering is determined by the classification matrix. They are equal if the number
    /// of connections to the $i$-th class is the same for both nodes. Otherwise, their ordering is the
    /// lexicographical ordering of their number of connections to the classes.
    fn compare_node_classification(&self, first_node: usize, second_node: usize) -> Ordering {
        return match self.find_class(first_node).cmp(&self.find_class(second_node)) {
            Ordering::Equal => {
                for (x, y) in izip!(&self.matrix[first_node], &self.matrix[second_node]) {
                    match x.cmp(y) {
                        Ordering::Equal => (),
                        ord => return ord.reverse(),
                    }
                }
                return Ordering::Equal;
            }
            Ordering::Less => Ordering::Equal,
            Ordering::Greater => Ordering::Greater,
        };
    }

    /// Construct a new classification from the current one from `adjacency_matrix`. The nodes are always
    /// classified such, that nodes $N_1$ and $N_2$ with $N_1 = N_2$ according to [compare_node_classification]
    /// are in the same class. If $N_1 < N_2$, the class containing $N_1$ and $N_2$ is split such, that they are
    /// in different classes. If there is a node pair with $N_1 > N_2$, the connections in `adjacency_matrix`
    /// are inconsistent with the classification, and `adjacency_matrix` is rejected by returning `None`.
    pub(crate) fn update_classification(&self, adjacency_matrix: &SymmetricMatrix) -> Option<Self> {
        let mut classification = (*self).clone();
        let mut new_boundaries: Vec<(usize, usize)> = Vec::new();
        let mut rerun = true;
        while rerun {
            rerun = false;
            for class in 0..classification.n_classes() {
                if classification.boundaries[class + 1] - classification.boundaries[class] == 1 {
                    continue;
                }
                for node in classification.boundaries[class]..(classification.boundaries[class + 1] - 1) {
                    match classification.compare_node_classification(node, node + 1) {
                        Ordering::Equal => (),
                        Ordering::Less => {
                            new_boundaries.push((class + 1, node + 1));
                            rerun = true;
                        }
                        Ordering::Greater => return None,
                    }
                }
            }
            for (i, (class, node)) in new_boundaries.drain(..).enumerate() {
                classification.boundaries.insert(class + i, node);
            }
            classification.update_classification_matrix(adjacency_matrix);
        }
        return Some(classification);
    }
}
