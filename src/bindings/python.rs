use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PySyntaxError};
use std::path::PathBuf;
use crate::{
    model::{
        ModelError, Model, TopologyModel
    }, 
    topology::{
        Topology, TopologyContainer, TopologyGenerator, filter::TopologySelector
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
    return Ok(());
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

    fn __str(&self) -> String {
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

    fn __str(&self) -> String {
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

    fn clear_critera(&mut self) {
        self.0.clear_criteria();
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str(&self) -> String {
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
#[pyo3(name = "Topology")]
struct PyTopology(Topology);

#[pymethods]
impl PyTopology {
    fn get_adjacency_matrix(&self) -> Vec<Vec<usize>> {
        return self.0.get_adjacency_matrix();
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str(&self) -> String {
        return format!("{:?}", self.0);
    }
}

#[pyclass]
#[pyo3(name = "TopologyContainer")]
struct PyTopologyContainer(TopologyContainer);

#[pymethods]
impl PyTopologyContainer {
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
        if let Some(selector) = selector {
            return Self(TopologyGenerator::new(n_external, n_loops, model.into(), Some(selector.into())));
        } else {
            return Self(TopologyGenerator::new(n_external, n_loops, model.into(), None));
        }
    }

    fn generate(&self) -> PyTopologyContainer {
        return PyTopologyContainer(self.0.generate());
    }
}