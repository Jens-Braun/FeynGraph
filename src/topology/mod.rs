use crate::topology::components::TopologyWorkspace;
use crate::topology::matrix::SymmetricMatrix;
use crate::util::factorial;

pub mod matrix;
pub mod components;

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
        }
        return Topology {
            adjacency_matrix: workspace.adjacency_matrix.clone(),
            node_symmetry,
            edge_symmetry
        }
    }
}