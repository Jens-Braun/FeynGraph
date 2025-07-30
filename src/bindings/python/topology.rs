use crate::{
    model::TopologyModel,
    topology::{Edge, Node, Topology, TopologyContainer, TopologyGenerator, filter::TopologySelector},
};
use itertools::Itertools;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyFunction;
use std::sync::Arc;

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "TopologyModel")]
pub(super) struct PyTopologyModel(pub(super) TopologyModel);

#[pymethods]
impl PyTopologyModel {
    #[new]
    pub(super) fn new(degrees: Vec<usize>) -> Self {
        return Self(TopologyModel::from(degrees));
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

impl From<PyTopologyModel> for TopologyModel {
    fn from(py_model: PyTopologyModel) -> Self {
        return py_model.0;
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "TopologySelector")]
pub(super) struct PyTopologySelector(TopologySelector);

#[pymethods]
impl PyTopologySelector {
    #[new]
    pub(super) fn new() -> Self {
        return Self(TopologySelector::default());
    }

    fn select_node_degree(&mut self, degree: usize, selection: usize) {
        self.0.add_node_degree(degree, selection);
    }

    fn select_node_degree_range(&mut self, degree: usize, start: usize, end: usize) {
        self.0.add_node_degree_range(degree, start..end);
    }

    fn select_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.0.add_node_partition(partition);
    }

    fn select_opi_components(&mut self, opi_count: usize) {
        self.0.add_opi_count(opi_count);
    }

    pub(super) fn add_custom_function(&mut self, py_function: Py<PyFunction>) {
        self.0.add_custom_function(Arc::new(move |topo: &Topology| -> bool {
            Python::with_gil(|py| -> bool {
                py_function
                    .call1(py, (PyTopology(topo.clone()),))
                    .unwrap()
                    .extract(py)
                    .unwrap()
            })
        }))
    }

    fn select_on_shell(&mut self) {
        self.0.set_on_shell();
    }

    fn select_self_loops(&mut self, n: usize) {
        self.0.add_self_loop_count(n);
    }

    fn clear(&mut self) {
        self.0.clear_criteria();
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

impl From<PyTopologySelector> for TopologySelector {
    fn from(selector: PyTopologySelector) -> Self {
        return selector.0;
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Node")]
struct PyNode(Node);

#[pymethods]
impl PyNode {
    fn nodes(&self) -> Vec<usize> {
        return self.0.adjacent_nodes.clone();
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Edge")]
struct PyEdge(Edge);

#[pymethods]
impl PyEdge {
    fn nodes(&self) -> [usize; 2] {
        return self.0.connected_nodes;
    }

    fn momentum(&self) -> Vec<i8> {
        return self.0.momenta.as_ref().unwrap().clone();
    }
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Topology")]
pub(super) struct PyTopology(pub(super) Topology);

#[pymethods]
impl PyTopology {
    fn nodes(&self) -> Vec<PyNode> {
        return self.0.nodes_iter().map(|node| PyNode(node.clone())).collect_vec();
    }

    fn edges(&self) -> Vec<PyEdge> {
        return self.0.edges_iter().map(|edge| PyEdge(edge.clone())).collect_vec();
    }

    fn symmetry_factor(&self) -> usize {
        return self.0.node_symmetry * self.0.edge_symmetry;
    }

    fn draw_tikz(&self, path: String) -> PyResult<()> {
        self.0.draw_tikz(path)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn _repr_svg_(&self) -> String {
        return self.0.draw_svg();
    }

    fn __str__(&self) -> String {
        return format!("{}", self.0);
    }
}

#[pyclass]
#[pyo3(name = "TopologyContainer")]
pub(super) struct PyTopologyContainer(TopologyContainer);

#[pymethods]
impl PyTopologyContainer {
    fn query(&self, selector: &PyTopologySelector) -> Option<usize> {
        return self.0.query(&selector.0);
    }

    #[pyo3(signature = (topologies, n_cols = None))]
    fn draw(&self, topologies: Vec<usize>, n_cols: Option<usize>) -> String {
        return self.0.draw_svg(&topologies, n_cols);
    }

    pub(super) fn __len__(&self) -> usize {
        return self.0.len();
    }
    fn __getitem__(&self, i: usize) -> PyResult<PyTopology> {
        if i >= self.0.len() {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        return Ok(PyTopology((*self.0.get(i)).clone()));
    }
}

#[pyclass]
#[pyo3(name = "TopologyGenerator")]
pub(super) struct PyTopologyGenerator(TopologyGenerator);

#[pymethods]
impl PyTopologyGenerator {
    #[new]
    #[pyo3(signature = (n_external, n_loops, model, selector=None))]
    pub(super) fn new(
        n_external: usize,
        n_loops: usize,
        model: PyTopologyModel,
        selector: Option<PyTopologySelector>,
    ) -> PyTopologyGenerator {
        return if let Some(selector) = selector {
            Self(TopologyGenerator::new(
                n_external,
                n_loops,
                model.into(),
                Some(selector.into()),
            ))
        } else {
            Self(TopologyGenerator::new(n_external, n_loops, model.into(), None))
        };
    }

    pub(super) fn generate(&self, py: Python<'_>) -> PyTopologyContainer {
        return py.allow_threads(|| -> PyTopologyContainer {
            return PyTopologyContainer(self.0.generate());
        });
    }
}
