use itertools::Itertools;
use pyo3::{prelude::*, pyfunction, pymethods};

use super::python::topology::{PyTopology, PyTopologyContainer};
use crate::{
    diagram::{DiagramGenerator, view::DiagramView},
    model::{Model, TopologyModel},
    topology::{Topology, TopologyGenerator},
};
use std::fmt::Write;

const fn feynarts_particle_map(pdg: isize) -> &'static str {
    match pdg {
        1 => "F[4, {1}]",
        2 => "F[3, {1}]",
        3 => "F[4, {2}]",
        4 => "F[3, {2}]",
        5 => "F[4, {3}]",
        6 => "F[3, {3}]",
        11 => "F[2, {1}]",
        12 => "F[1, {1}]",
        13 => "F[2, {2}]",
        14 => "F[1, {2}]",
        15 => "F[2, {3}]",
        16 => "F[1, {3}]",
        -1 => "-F[4, {1}]",
        -2 => "-F[3, {1}]",
        -3 => "-F[4, {2}]",
        -4 => "-F[3, {2}]",
        -5 => "-F[4, {3}]",
        -6 => "-F[3, {3}]",
        -11 => "-F[2, {1}]",
        -12 => "-F[1, {1}]",
        -13 => "-F[2, {2}]",
        -14 => "-F[1, {2}]",
        -15 => "-F[2, {3}]",
        -16 => "-F[1, {3}]",
        21 => "V[5]",
        22 => "V[1]",
        23 => "V[2]",
        24 => "-V[3]",
        -24 => "V[3]",
        25 => "S[1]",
        9000001 => "U[1]",
        9000002 => "U[2]",
        9000003 => "U[4]",
        9000004 => "U[3]",
        82 => "U[5]",
        -9000001 => "-U[1]",
        -9000002 => "-U[2]",
        -9000003 => "-U[4]",
        -9000004 => "-U[3]",
        -82 => "-U[5]",
        250 => "S[2]",
        251 => "-S[3]",
        -251 => "S[3]",
        _ => "Unknown PDG",
    }
}

#[pyfunction]
#[pyo3(name = "_diagrams_feynarts")]
pub(crate) fn diagrams_feynarts(
    py: Python<'_>,
    particles_in: Vec<String>,
    particles_out: Vec<String>,
    n_loops: usize,
) -> PyResult<String> {
    let mut buffer = String::new();
    let sm = Model::default();
    let n_in = particles_in.len();
    py.allow_threads(|| -> PyResult<()> {
        let topos = TopologyGenerator::new(
            particles_in.len() + particles_out.len(),
            n_loops,
            TopologyModel::from(vec![3, 4]),
            None,
        )
        .generate();
        write!(
            buffer,
            "TopologyList[Process -> {{{}}} -> {{{}}}, Model -> \"SMQCD\", InsertionLevel -> {{Particles}}][",
            particles_in
                .iter()
                .map(|p| feynarts_particle_map(sm.get_particle_by_name(p).unwrap().pdg_code))
                .join(", "),
            particles_out
                .iter()
                .map(|p| feynarts_particle_map(sm.get_particle_by_name(p).unwrap().pdg_code))
                .join(", ")
        )
        .unwrap();
        let generator = DiagramGenerator::new(
            &particles_in.iter().map(String::as_ref).collect_vec(),
            &particles_out.iter().map(String::as_ref).collect_vec(),
            n_loops,
            sm,
            None,
        )?;
        for topo in topos.iter() {
            let diags = generator.assign_topology(topo).unwrap();
            if diags.len() == 0 {
                continue;
            }
            write!(
                buffer,
                "{} -> Insertions[Particles][",
                topo.to_feynarts(Some(n_in), true)
            )
            .unwrap();
            for (j, d) in diags.views().enumerate() {
                buffer.push_str(&d.to_feynarts(j + 1));
                if j != diags.len() - 1 {
                    buffer.push(',');
                }
            }
            write!(buffer, "],").unwrap();
        }
        buffer.pop(); // Remove final ','
        write!(buffer, "]").unwrap();
        return Ok(());
    })?;
    return Ok(buffer);
}

impl Topology {
    pub(crate) fn to_feynarts(&self, n_in: Option<usize>, include_fields: bool) -> String {
        let mut buffer = String::new();
        write!(buffer, "Topology[{}][", self.edge_symmetry * self.node_symmetry).unwrap();
        for (i, edge) in self.edges_iter().enumerate() {
            write!(
                buffer,
                "Propagator[{}][Vertex[{}][{}], Vertex[{}][{}]{}]",
                if edge.connected_nodes[0] < self.n_external {
                    if let Some(n_in) = n_in {
                        if edge.connected_nodes[0] < n_in {
                            "Incoming"
                        } else {
                            "Outgoing"
                        }
                    } else {
                        "External"
                    }
                } else if edge
                    .momenta
                    .as_ref()
                    .unwrap()
                    .iter()
                    .skip(self.n_external)
                    .any(|x| *x != 0)
                {
                    "FALoop[1]"
                } else {
                    "Internal"
                },
                self.nodes[edge.connected_nodes[0]].degree,
                edge.connected_nodes[0] + 1,
                self.nodes[edge.connected_nodes[1]].degree,
                edge.connected_nodes[1] + 1,
                if include_fields {
                    format!(", Field[{}]", i + 1)
                } else {
                    "".into()
                }
            )
            .unwrap();
            if i != self.n_edges() - 1 {
                write!(buffer, ", ").unwrap()
            }
        }
        write!(buffer, "]").unwrap();
        return buffer;
    }
}

#[pymethods]
impl PyTopology {
    #[pyo3(signature = (n_in = None))]
    fn to_feynarts(&self, n_in: Option<usize>) -> String {
        return self.0.to_feynarts(n_in, false);
    }
}

#[pymethods]
impl PyTopologyContainer {
    #[pyo3(signature = (n_in = None))]
    pub(crate) fn to_feynarts(&self, n_in: Option<usize>) -> String {
        let mut buffer = String::new();
        write!(buffer, "TopologyList[").unwrap();
        for (i, topo) in self.0.data.iter().enumerate() {
            buffer.push_str(&topo.to_feynarts(n_in, false));
            if i != self.0.data.len() - 1 {
                buffer.push(',');
            }
        }
        write!(buffer, "]").unwrap();
        return buffer;
    }
}

impl DiagramView<'_> {
    fn to_feynarts(&self, id: usize) -> String {
        let mut buffer = String::new();
        let mut i = 0;
        write!(buffer, "FeynmanGraph[{}, Particles == {}][", self.symmetry_factor(), id).unwrap();
        for leg in self.incoming().chain(self.outgoing()) {
            write!(
                buffer,
                "Field[{}] -> {},",
                i + 1,
                feynarts_particle_map(leg.particle().pdg())
            )
            .unwrap();
            i += 1;
        }
        for prop in self.propagators() {
            write!(
                buffer,
                "Field[{}] -> {},",
                i + 1,
                feynarts_particle_map(prop.particle().pdg())
            )
            .unwrap();
            i += 1;
        }
        buffer.pop(); // Remove last ','
        write!(buffer, "]").unwrap();
        return buffer;
    }
}
