use pyo3::{
    exceptions::{PyIOError, PySyntaxError, PyValueError},
    prelude::*,
};

use feyngraph_core::InputError;
use feyngraph_model::ModelError;

use model::{PyInteraction, PyModel, PyParticle};

use crate::diagrams::{PyDiagramContainer, PyDiagramGenerator, PyDiagramSelector};

pub(crate) mod diagrams;
#[cfg(feature = "drawing")]
pub(crate) mod drawing;
pub(crate) mod model;
pub(crate) mod topology;

#[pymodule]
mod feyngraph {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
    }

    #[pymodule_export]
    use crate::generate_diagrams;

    #[pymodule_export]
    use crate::diagrams::{
        PyDiagram, PyDiagramContainer, PyDiagramGenerator, PyDiagramSelector, PyLeg, PyPropagator, PyVertex,
    };

    #[pymodule_export]
    use crate::model::{ModelKind, PyInteraction, PyModel, PyParticle};

    #[pymodule]
    mod topology {
        #[pymodule_export]
        use crate::topology::{
            PyEdge, PyNode, PyTopology, PyTopologyContainer, PyTopologyGenerator, PyTopologyModel, PyTopologySelector,
        };
    }

    #[cfg(feature = "drawing")]
    #[pymodule]
    mod drawing {
        #[pymodule_export]
        use crate::drawing::{PyColor, PyDecoration, PyDecorationKind, PyPathStyle, PyStroke, PyTheme};
    }
}

#[pyfunction]
#[pyo3(signature = (
    particles_in,
    particles_out,
    n_loops = 0,
    model = PyModel::__new__(),
    selector = None,
    threads = 0,
))]
fn generate_diagrams(
    py: Python<'_>,
    particles_in: Vec<String>,
    particles_out: Vec<String>,
    n_loops: usize,
    model: PyModel,
    selector: Option<PyDiagramSelector>,
    threads: usize,
) -> PyResult<PyDiagramContainer> {
    let diagram_selector;
    if let Some(in_selector) = selector {
        diagram_selector = in_selector;
    } else {
        diagram_selector = PyDiagramSelector::new();
    }
    return Ok(
        PyDiagramGenerator::new(particles_in, particles_out, n_loops, model, Some(diagram_selector))?
            .generate(py, threads)?,
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
        match err.0.kind {
            feyngraph_model::ErrorKind::IO(_) => PyIOError::new_err(err.0.to_string()),
            _ => PySyntaxError::new_err(err.0.to_string()),
        }
    }
}

impl From<PyErr> for PyModelError {
    fn from(value: PyErr) -> Self {
        PyModelError(ModelError::misc(format!("Python runtime error: {value}"), value))
    }
}

impl From<PyModelError> for ModelError {
    fn from(value: PyModelError) -> Self {
        value.0
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

#[cfg(test)]
mod tests {
    use crate::topology::{PyTopologyGenerator, PyTopologyModel, PyTopologySelector};

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
