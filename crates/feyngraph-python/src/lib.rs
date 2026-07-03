use log::warn;
use pyo3::{
    exceptions::{PyIOError, PySyntaxError, PyValueError},
    prelude::*,
};
use std::{error::Error, path::PathBuf};

use crate::{
    diagrams::{PyDiagram, PyDiagramContainer, PyDiagramGenerator, PyDiagramSelector, PyLeg, PyPropagator, PyVertex},
    topology::{PyTopology, PyTopologyContainer, PyTopologyGenerator, PyTopologyModel, PyTopologySelector},
};
use feyngraph_core::{
    InputError,
    model::{InteractionVertex, LineStyle, Model, ModelError, Particle, Statistic, TopologyModel},
};

use indexmap::IndexMap as OriginalIndexMap;
use rustc_hash::{FxBuildHasher, FxHashMap};

pub(crate) type IndexMap<K, V> = OriginalIndexMap<K, V, FxBuildHasher>;
pub(crate) type HashMap<K, V> = FxHashMap<K, V>;

#[cfg(feature = "drawing")]
use drawing::{PyColor, PyDecoration, PyDecorationKind, PyPathStyle, PyStroke, PyTheme};

pub(crate) mod diagrams;
#[cfg(feature = "drawing")]
pub(crate) mod drawing;
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
    #[cfg(feature = "drawing")]
    {
        let drawing_submodule = PyModule::new(m.py(), "drawing")?;
        drawing_submodule.add_class::<PyColor>()?;
        drawing_submodule.add_class::<PyDecoration>()?;
        drawing_submodule.add_class::<PyDecorationKind>()?;
        drawing_submodule.add_class::<PyPathStyle>()?;
        drawing_submodule.add_class::<PyStroke>()?;
        drawing_submodule.add_class::<PyTheme>()?;
        m.add_submodule(&drawing_submodule)?;
    }
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

struct PyModelError(ModelError);

impl From<ModelError> for PyModelError {
    fn from(err: ModelError) -> Self {
        PyModelError(err)
    }
}

impl From<PyModelError> for PyErr {
    fn from(err: PyModelError) -> PyErr {
        match err.0 {
            ModelError::IOError(_, _) => PyIOError::new_err(err.0.to_string()),
            ModelError::ParseError(_, _) => PySyntaxError::new_err(err.0.to_string()),
            ModelError::ContentError(_) => PySyntaxError::new_err(err.0.to_string()),
        }
    }
}

struct PyInputError(InputError);

impl From<InputError> for PyInputError {
    fn from(err: InputError) -> Self {
        PyInputError(err)
    }
}

impl From<PyInputError> for PyErr {
    fn from(err: PyInputError) -> PyErr {
        PyValueError::new_err(err.0.to_string())
    }
}

#[derive(Clone)]
#[pyclass(from_py_object)]
#[pyo3(name = "Model")]
pub(crate) struct PyModel(Model);

#[pymethods]
impl PyModel {
    #[new]
    fn __new__() -> Self {
        return PyModel(Model::default());
    }

    #[staticmethod]
    fn from_ufo(path: PathBuf) -> Result<Self, PyModelError> {
        return Ok(Self(Model::from_ufo(&path)?));
    }

    #[staticmethod]
    fn from_qgraf(path: PathBuf) -> Result<Self, PyModelError> {
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
    ) -> Result<(), PyModelError> {
        self.0.add_vertex(name, particles, spin_map, coupling_orders)?;
        Ok(())
    }

    fn merge_vertices(&mut self) -> IndexMap<String, Vec<String>> {
        return self.0.merge_vertices();
    }

    fn add_coupling(&mut self, vertex: String, coupling: String, power: usize) -> Result<(), PyModelError> {
        self.0.add_coupling(vertex, coupling, power)?;
        Ok(())
    }

    fn split_vertex(&mut self, vertex: String, new_vertices: Vec<String>) -> Result<(), PyModelError> {
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
#[pyclass(skip_from_py_object)]
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
#[pyclass(skip_from_py_object)]
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
        return self.0.coupling_orders().clone();
    }

    fn name(&self) -> String {
        return self.0.name().to_owned();
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
