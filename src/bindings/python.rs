use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PySyntaxError};
use std::path::PathBuf;
use std::sync::Arc;
use itertools::Itertools;
use pyo3::types::PyFunction;
use crate::{
    model::{
        ModelError, Model, TopologyModel
    }, 
    topology::{
        Node, Edge, Topology, TopologyContainer, TopologyGenerator, 
        filter::{TopologySelector, SelectionCriterion}
    }
};

#[pymodule]
#[allow(non_snake_case)]
fn feyngraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    let topology_submodule = PyModule::new(m.py(), "topology")?;
    topology_submodule.add_class::<PyTopologyModel>()?;
    topology_submodule.add_class::<PyTopology>()?;
    topology_submodule.add_class::<PyTopologyContainer>()?;
    topology_submodule.add_class::<PyTopologyGenerator>()?;
    topology_submodule.add_class::<PyTopologySelector>()?;
    m.add_submodule(&topology_submodule)?;
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    return Ok(());
}

#[pyfunction]
fn set_threads(n_threads: usize) {
    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global().unwrap();
}

impl std::convert::From<ModelError> for PyErr {
    fn from(err: ModelError) -> PyErr {
        match err {
            ModelError::IOError(_) => PyIOError::new_err(err.to_string()),
            ModelError::ParseError(_) => PySyntaxError::new_err(err.to_string()),
            ModelError::ContentError(_) => PySyntaxError::new_err(err.to_string()),
        }
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "TopologyModel")]
struct PyTopologyModel(TopologyModel);

#[pymethods]
impl PyTopologyModel {
    #[new]
    fn new(degrees: Vec<usize>) -> Self {
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
#[pyo3(name = "Model")]
struct PyModel(Model);

#[pymethods]
impl PyModel {
    #[staticmethod]
    fn from_ufo(path: PathBuf) -> PyResult<Self> {
        return Ok(Self(Model::from_ufo(&path)?));
    }

    fn as_topology_model(&self) -> PyTopologyModel {
        return PyTopologyModel(TopologyModel::from(&self.0));
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

impl From<PyModel> for Model {
    fn from(py_model: PyModel) -> Self {
        return py_model.0;
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "TopologySelector")]
struct PyTopologySelector(TopologySelector);

#[pymethods]
impl PyTopologySelector {
    #[new]
    fn new() -> Self {
        return Self(TopologySelector::default());
    }
    
    #[staticmethod]
    fn empty() -> Self {
        return Self(TopologySelector::new());
    }

    fn select_node_degree(&mut self, degree: usize, selection: usize) {
        self.0.add_node_degree(degree, selection);
    }

    fn add_node_degree_range(&mut self, degree: usize, start: usize, end: usize) {
        self.0.add_node_degree_range(degree, start..end);
    }

    fn select_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.0.add_node_partition(partition);
    }

    fn select_opi_components(&mut self, opi_count: usize) {
        self.0.add_opi_count(opi_count);
    }
    
    fn add_custom_function(&mut self, py_function: Py<PyFunction>) {
        self.0.add_criterion(SelectionCriterion::CustomCriterion(
            Arc::new(move |topo: &Topology| -> bool { 
                Python::with_gil( |py| -> bool {
                    py_function.call1(py, (PyTopology(topo.clone()),)).unwrap().extract(py).unwrap()
                }
                )
                
            })
        ))
    }

    fn clear_critera(&mut self) {
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
    
    fn get_nodes(&self) -> (usize, usize) {
        return self.0.connected_nodes.clone();
    }
    
    fn get_momentum(&self) -> Vec<i8> {
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
struct PyTopology(Topology);

#[pymethods]
impl PyTopology {
    fn get_nodes(&self) -> Vec<PyNode> {
        return self.0.nodes_iter().map(|node| PyNode(node.clone())).collect_vec();
    }

    fn get_edges(&self) -> Vec<PyEdge> {
        return self.0.edges_iter().map(|edge| PyEdge(edge.clone())).collect_vec();
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{}", self.0);
    }
}

#[pyclass]
#[pyo3(name = "TopologyContainer")]
struct PyTopologyContainer(TopologyContainer);

#[pymethods]
impl PyTopologyContainer {
    
    fn query(&self, selector: &PyTopologySelector) -> Option<usize> {
        return self.0.query(&selector.0);
    }
    
    fn __len__(&self) -> usize {
        return self.0.len();
    }
    fn __getitem__(&self, i: usize) -> PyTopology {
        return PyTopology((*self.0.get(i)).clone());
    }
}

#[pyclass]
#[pyo3(name = "TopologyGenerator")]
struct PyTopologyGenerator(TopologyGenerator);

#[pymethods]
impl PyTopologyGenerator {
    #[new]
    #[pyo3(signature = (n_external, n_loops, model, selector=None))]
    fn new(n_external: usize, 
        n_loops: usize, 
        model: PyTopologyModel, 
        selector: Option<PyTopologySelector>
    ) -> PyTopologyGenerator {
        return if let Some(selector) = selector {
            Self(TopologyGenerator::new(n_external, n_loops, model.into(), Some(selector.into())))
        } else {
            Self(TopologyGenerator::new(n_external, n_loops, model.into(), None))
        }
    }

    fn generate(&self, py: Python<'_>) -> PyTopologyContainer {
        return py.allow_threads(
            || -> PyTopologyContainer {
                return PyTopologyContainer(self.0.generate());
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::PyFunction;
    use pyo3_ffi::c_str;
    use super::*;

    #[test]
    fn py_topology_generator_py_function() {
        let filter: Py<PyFunction> = Python::with_gil(|py| -> Py<PyFunction> {
           PyModule::from_code(py, c_str!("def no_self_loops(topo):
    for edge in topo.get_edges():
        nodes = edge.get_nodes()
        if nodes[0] == nodes[1]:
            return False
    return True
           "),
           c_str!(""),
           c_str!("")
           ).unwrap()
               .getattr("no_self_loops")
               .unwrap()
               .downcast_into().unwrap()
               .unbind()
        });
        let mut selector = PyTopologySelector::new();
        selector.add_custom_function(filter);
        let generator = PyTopologyGenerator::new(
            2, 
            1, 
            PyTopologyModel::new(vec![3, 4]),
            Some(selector)
        );
        let topologies = Python::with_gil(|py| {
            generator.generate(py)
        });        
        assert_eq!(topologies.__len__(), 1);
    }
}