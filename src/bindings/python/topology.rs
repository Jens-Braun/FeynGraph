use crate::{
    model::TopologyModel,
    topology::{Edge, Node, Topology, TopologyContainer, TopologyGenerator, filter::TopologySelector},
};
use itertools::Itertools;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "TopologyModel")]
pub(crate) struct PyTopologyModel(pub(crate) TopologyModel);

#[pymethods]
impl PyTopologyModel {
    #[new]
    pub(crate) fn new(degrees: Vec<usize>) -> Self {
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
pub(crate) struct PyTopologySelector(TopologySelector);

#[pymethods]
impl PyTopologySelector {
    #[new]
    pub(crate) fn new() -> Self {
        return Self(TopologySelector::default());
    }

    fn select_node_degree(&mut self, degree: usize, selection: usize) {
        self.0.select_node_degree(degree, selection);
    }

    fn select_node_degree_range(&mut self, degree: usize, start: usize, end: usize) {
        self.0.select_node_degree_range(degree, start..end);
    }

    fn select_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.0.select_node_partition(partition);
    }

    fn select_opi_components(&mut self, opi_count: usize) {
        self.0.select_opi_components(opi_count);
    }

    pub(crate) fn add_custom_function(&mut self, py_function: Py<PyAny>) {
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
        self.0.select_on_shell();
    }

    fn select_self_loops(&mut self, n: usize) {
        self.0.select_self_loops(n);
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
pub(crate) struct PyNode(Node);

#[pymethods]
impl PyNode {
    pub(crate) fn adjacent(&self) -> Vec<usize> {
        return self.0.adjacent_nodes.clone();
    }

    fn degree(&self) -> usize {
        return self.0.degree;
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
pub(crate) struct PyEdge(Edge);

#[pymethods]
impl PyEdge {
    pub(crate) fn nodes(&self) -> [usize; 2] {
        return self.0.connected_nodes;
    }

    pub(crate) fn momentum(&self) -> Vec<i8> {
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
pub(crate) struct PyTopology(pub(crate) Topology);

#[pymethods]
impl PyTopology {
    pub(crate) fn nodes(&self) -> Vec<PyNode> {
        return self.0.nodes_iter().map(|node| PyNode(node.clone())).collect_vec();
    }

    pub(crate) fn edges(&self) -> Vec<PyEdge> {
        return self.0.edges_iter().map(|edge| PyEdge(edge.clone())).collect_vec();
    }

    pub(crate) fn symmetry_factor(&self) -> usize {
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
        return self.0.draw_svg_string();
    }

    fn __str__(&self) -> String {
        return format!("{}", self.0);
    }
}

#[pyclass]
#[pyo3(name = "TopologyContainer")]
pub(crate) struct PyTopologyContainer(pub(crate) TopologyContainer);

#[pymethods]
impl PyTopologyContainer {
    fn query(&self, selector: &PyTopologySelector) -> Option<usize> {
        return self.0.query(&selector.0);
    }

    #[pyo3(signature = (topologies, n_cols = None))]
    fn draw(&self, topologies: Vec<usize>, n_cols: Option<usize>) -> String {
        return self.0.draw_svg(&topologies, n_cols);
    }

    pub(crate) fn __len__(&self) -> usize {
        return self.0.len();
    }

    fn __getitem__(&self, i: usize) -> PyResult<PyTopology> {
        if i >= self.0.len() {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        return Ok(PyTopology((*self.0.get(i)).clone()));
    }

    fn _repr_svg_(&self) -> String {
        let n = self.0.len().min(100);
        return self.0.draw_svg(&(0..n).collect_vec(), None);
    }
}

#[pyclass]
#[pyo3(name = "TopologyGenerator")]
pub(crate) struct PyTopologyGenerator(TopologyGenerator);

#[pymethods]
impl PyTopologyGenerator {
    #[new]
    #[pyo3(signature = (n_external, n_loops, model, selector=None))]
    pub(crate) fn new(
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

    pub(crate) fn generate(&self, py: Python<'_>) -> PyTopologyContainer {
        return py.allow_threads(|| -> PyTopologyContainer {
            return PyTopologyContainer(self.0.generate());
        });
    }

    pub(crate) fn count(&self, py: Python<'_>) -> usize {
        return py.allow_threads(|| -> usize {
            return self.0.count_topologies();
        });
    }
}
