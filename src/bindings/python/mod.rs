#[cfg(feature = "wolfram-bindings")]
use super::wolfram::diagrams_feynarts;
use crate::model::{LineStyle, Statistic};
use crate::util::{HashMap, IndexMap};
use crate::{
    model::{InteractionVertex, Model, ModelError, Particle, TopologyModel},
    util,
};
use diagrams::{PyDiagram, PyDiagramContainer, PyDiagramGenerator, PyDiagramSelector, PyLeg, PyPropagator, PyVertex};
use log::warn;
use pyo3::exceptions::{PyIOError, PySyntaxError, PyValueError};
use pyo3::prelude::*;
use std::error::Error;
use std::path::PathBuf;
use topology::{PyTopology, PyTopologyContainer, PyTopologyGenerator, PyTopologyModel, PyTopologySelector};

pub(crate) mod diagrams;
pub(crate) mod topology;

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
    m.add_class::<PyDiagram>()?;
    m.add_class::<PyDiagramGenerator>()?;
    m.add_class::<PyDiagramContainer>()?;
    m.add_class::<PyDiagramSelector>()?;
    m.add_class::<PyPropagator>()?;
    m.add_class::<PyLeg>()?;
    m.add_class::<PyVertex>()?;
    m.add_class::<PyParticle>()?;
    m.add_class::<PyInteractionVertex>()?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(generate_diagrams, m)?)?;
    #[cfg(feature = "wolfram-bindings")]
    m.add("_WOLFRAM_ENABLED", true)?;
    #[cfg(not(feature = "wolfram-bindings"))]
    m.add("_WOLFRAM_ENABLED", false)?;
    #[cfg(feature = "wolfram-bindings")]
    m.add_function(wrap_pyfunction!(diagrams_feynarts, m)?)?;
    return Ok(());
}

#[pyfunction]
fn set_threads(n_threads: usize) -> PyResult<()> {
    return match rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global() {
        Ok(()) => Ok(()),
        Err(e) => match e.source() {
            None => {
                warn!(
                    "The Rayon thread pool has already been initialized, which is only possible once. This call will be ignored."
                );
                Ok(())
            }
            Some(e) => Err(PyIOError::new_err(format!(
                "Error while initializing the global Rayon thread pool: {}",
                e
            ))),
        },
    };
}

#[pyfunction]
#[pyo3(signature = (
    particles_in,
    particles_out,
    n_loops = 0,
    model = PyModel::__new__(),
    selector = None,
))]
fn generate_diagrams(
    py: Python<'_>,
    particles_in: Vec<String>,
    particles_out: Vec<String>,
    n_loops: usize,
    model: PyModel,
    selector: Option<PyDiagramSelector>,
) -> PyResult<PyDiagramContainer> {
    let diagram_selector;
    if let Some(in_selector) = selector {
        diagram_selector = in_selector;
    } else {
        diagram_selector = PyDiagramSelector::new();
    }
    return Ok(
        PyDiagramGenerator::new(particles_in, particles_out, n_loops, model, Some(diagram_selector))?.generate(py),
    );
}

impl From<ModelError> for PyErr {
    fn from(err: ModelError) -> PyErr {
        match err {
            ModelError::IOError(_, _) => PyIOError::new_err(err.to_string()),
            ModelError::ParseError(_, _) => PySyntaxError::new_err(err.to_string()),
            ModelError::ContentError(_) => PySyntaxError::new_err(err.to_string()),
        }
    }
}

impl From<util::Error> for PyErr {
    fn from(err: util::Error) -> PyErr {
        match err {
            util::Error::InputError(_) => PyValueError::new_err(err.to_string()),
        }
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Model")]
pub(crate) struct PyModel(Model);

#[pymethods]
impl PyModel {
    #[new]
    fn __new__() -> Self {
        return PyModel(Model::default());
    }

    #[staticmethod]
    fn from_ufo(path: PathBuf) -> PyResult<Self> {
        return Ok(Self(Model::from_ufo(&path)?));
    }

    #[staticmethod]
    fn from_qgraf(path: PathBuf) -> PyResult<Self> {
        return Ok(Self(Model::from_qgraf(&path)?));
    }

    #[staticmethod]
    fn empty() -> Self {
        return PyModel(Model::empty());
    }

    fn add_particle(
        &mut self,
        name: String,
        anti_name: String,
        spin: isize,
        color: isize,
        pdg_code: isize,
        texname: String,
        antitexname: String,
        linestyle: String,
        fermi: bool,
    ) -> PyResult<()> {
        let linestyle = match linestyle.to_lowercase().as_str() {
            "dashed" => LineStyle::Dashed,
            "dotted" => LineStyle::Dotted,
            "straight" => LineStyle::Straight,
            "wavy" => LineStyle::Wavy,
            "curly" => LineStyle::Curly,
            "scurly" => LineStyle::Scurly,
            "swavy" => LineStyle::Swavy,
            "double" => LineStyle::Double,
            "none" => LineStyle::None,
            _ => return Err(PySyntaxError::new_err(format!("Unknown line style '{linestyle}'"))),
        };
        self.0.add_particle(
            name,
            anti_name,
            pdg_code,
            spin,
            color,
            texname,
            antitexname,
            linestyle,
            if fermi { Statistic::Fermi } else { Statistic::Bose },
        );
        Ok(())
    }

    fn add_vertex(
        &mut self,
        name: String,
        particles: Vec<String>,
        spin_map: Vec<isize>,
        coupling_orders: HashMap<String, usize>,
    ) -> PyResult<()> {
        self.0.add_vertex(name, particles, spin_map, coupling_orders)?;
        Ok(())
    }

    fn merge_vertices(&mut self) -> IndexMap<String, Vec<String>> {
        return self.0.merge_vertices();
    }

    fn add_coupling(&mut self, vertex: String, coupling: String, power: usize) -> PyResult<()> {
        self.0.add_coupling(vertex, coupling, power)?;
        Ok(())
    }

    fn split_vertex(&mut self, vertex: String, new_vertices: Vec<String>) -> PyResult<()> {
        self.0.split_vertex(vertex, &new_vertices)?;
        Ok(())
    }

    fn as_topology_model(&self) -> PyTopologyModel {
        return PyTopologyModel(TopologyModel::from(&self.0));
    }

    fn particles(&self) -> Vec<PyParticle> {
        return self.0.particles_iter().map(|p| PyParticle(p.clone())).collect();
    }

    fn vertices(&self) -> Vec<PyInteractionVertex> {
        return self.0.vertices_iter().map(|v| PyInteractionVertex(v.clone())).collect();
    }

    fn splitting(&self, name: String) -> Option<HashMap<String, Vec<(usize, usize)>>> {
        return self.0.get_splitting(&name).cloned();
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
#[pyo3(name = "Particle")]
pub(crate) struct PyParticle(Particle);

#[pymethods]
impl PyParticle {
    pub(crate) fn name(&self) -> String {
        return self.0.name().clone();
    }

    pub(crate) fn anti_name(&self) -> String {
        return self.0.anti_name().clone();
    }

    pub(crate) fn is_anti(&self) -> bool {
        return self.0.is_anti();
    }

    pub(crate) fn is_fermi(&self) -> bool {
        return self.0.is_fermi();
    }

    pub(crate) fn self_anti(&self) -> bool {
        return self.0.self_anti();
    }

    pub(crate) fn pdg(&self) -> isize {
        return self.0.pdg();
    }

    pub(crate) fn spin(&self) -> isize {
        return self.0.spin();
    }

    pub(crate) fn color(&self) -> isize {
        return self.0.color();
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
#[pyo3(name = "InteractionVertex")]
pub(crate) struct PyInteractionVertex(InteractionVertex);

#[pymethods]
impl PyInteractionVertex {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }

    fn coupling_orders(&self) -> HashMap<String, usize> {
        return self.0.coupling_orders.clone();
    }

    fn name(&self) -> String {
        return self.0.name.clone();
    }

    fn order(&self, coupling: String) -> usize {
        return self.0.order(&coupling);
    }

    fn match_particles(&self, query: Vec<String>) -> bool {
        return self.0.match_particles(query.iter());
    }
}

impl From<PyModel> for Model {
    fn from(py_model: PyModel) -> Self {
        return py_model.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use pyo3_ffi::c_str;
    use test_log::test;

    #[test]
    fn py_topology_generator_py_function() {
        let filter: Py<PyAny> = Python::attach(|py| -> Py<PyAny> {
            PyModule::from_code(
                py,
                c_str!(
                    "def no_self_loops(topo):
    for edge in topo.edges():
        nodes = edge.nodes()
        if nodes[0] == nodes[1]:
            return False
    return True
           "
                ),
                c_str!(""),
                c_str!(""),
            )
            .unwrap()
            .getattr("no_self_loops")
            .unwrap()
            .unbind()
        });
        let mut selector = PyTopologySelector::new();
        selector.add_custom_function(filter);
        let generator = PyTopologyGenerator::new(2, 1, PyTopologyModel::new(vec![3, 4]), Some(selector));
        let topologies = Python::attach(|py| generator.generate(py));
        assert_eq!(topologies.__len__(), 1);
    }
}
