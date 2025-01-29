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
        filter::TopologySelector
    },
    diagram::{
        DiagramContainer, DiagramGenerator, Diagram, filter::DiagramSelector, Propagator, Vertex
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
    m.add_class::<PyDiagram>()?;
    m.add_class::<PyDiagramGenerator>()?;
    m.add_class::<PyDiagramContainer>()?;
    m.add_class::<PyDiagramSelector>()?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    return Ok(());
}

/// Set the number of threads FeynGraph will use. The default is the maximum number of available threads.
///
/// :param n_threads: number of threads to use
/// :type n_threads: int
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

/// A model containing only topological information, i.e. especially the allowed degrees of nodes.
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "TopologyModel")]
struct PyTopologyModel(TopologyModel);

#[pymethods]
impl PyTopologyModel {
    /// Create a new topology model containing nodes with degrees specified in `node_degrees`.
    ///
    /// :param degrees: included node degrees
    /// :type degrees: list[int]
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

/// Internal representation of a model in FeynGraph.
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Model")]
struct PyModel(Model);

#[pymethods]
impl PyModel {
    /// Import a model in the UFO format. The path should specify the path to the folder containing the model's `.py`
    /// files.
    ///
    /// :param path:
    /// :type path: str
    #[staticmethod]
    fn from_ufo(path: PathBuf) -> PyResult<Self> {
        return Ok(Self(Model::from_ufo(&path)?));
    }

    /// Cast the full model down to a topological model used by `TopologyGenerator`.
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

/// A selector class which determines whether a topology is to be kept or to be discarded. There are four types of
/// criteria available:
///
/// - node degrees: select only topologies for which the number of nodes with a specified degree matches any of the
/// given counts
/// - node partition: select only topologies matching any of the given node partitions, i.e. a topology for which the
/// number of nodes of each degree exactly matches the count specified in the partition.
/// - opi components: select only topologies for which the number of one-particle-irreducible components matches any of
/// the given counts
/// - custom functions: select only topologies for which any of the given custom functions return `true`
#[pymethods]
impl PyTopologySelector {
    /// Create a new topology selector, which contains no criteria and thus selects every diagram.
    #[new]
    fn new() -> Self {
        return Self(TopologySelector::default());
    }

    /// Add a constraint to only select topologies which contain `selection` nodes of degree `degree`.
    ///
    /// :param degree:
    /// :type degree: int
    /// :param selection:
    /// :type selection: int
    fn select_node_degree(&mut self, degree: usize, selection: usize) {
        self.0.add_node_degree(degree, selection);
    }

    /// Add a constraint to only select topologies which contain between `start` and `end` nodes of degree `degree`.
    ///
    /// :param degree: int
    /// :param start: int
    /// :param end: int
    /// :type degree: int
    /// :type start: int
    /// :type end: int
    fn select_node_degree_range(&mut self, degree: usize, start: usize, end: usize) {
        self.0.add_node_degree_range(degree, start..end);
    }

    /// Add a constraint to only select topologies for which the number of nodes of all given degree exactly matches
    /// he specified count.
    ///
    /// Examples
    /// --------
    /// .. code-block:: python
    ///
    ///     selector = TopologySelector()
    ///     # Select only topologies containing exactly four nodes of degree 3 and one node of degree 4
    ///     selector.select_node_partition([(3, 4), (4, 1)])
    ///
    /// :param partition:
    /// :type partition: list[tuple[int, int]]
    fn select_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.0.add_node_partition(partition);
    }

    /// Add a constraints to only select topologies with `opi_count` one-particle-irreducible components.
    ///
    /// :param opi_count:
    /// :type opi_count: int
    fn select_opi_components(&mut self, opi_count: usize) {
        self.0.add_opi_count(opi_count);
    }

    /// Add a constraint to only select topologies for which the given function returns `true`. The function receives
    /// a single topology as input and should return a boolean.
    ///
    /// Examples
    /// --------
    ///
    /// .. code-block:: python
    ///
    ///     def no_self_loop(topo: feyngraph.topology.Topology) -> bool:
    ///         return any(edge.get_nodes()[0] == edge.get_nodes()[1] for edge in topo.get_edges())
    ///
    ///     selector = feyngraph.topology.TopologySelector()
    ///     selector.add_custom_function(no_self_loop)
    ///
    /// :param py_function:
    /// :type py_function: Callable[[Topology], bool]
    fn add_custom_function(&mut self, py_function: Py<PyFunction>) {
        self.0.add_custom_function(
            Arc::new(move |topo: &Topology| -> bool {
                Python::with_gil( |py| -> bool {
                    py_function.call1(py, (PyTopology(topo.clone()),)).unwrap().extract(py).unwrap()
                }
                )
                
            })
        )
    }

    /// Clear all criteria.
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

/// The class representing a topological node.
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Node")]
struct PyNode(Node);

#[pymethods]
impl PyNode {

    /// Get the indices of all adjacent nodes.
    ///
    /// :return: list of indices of adjacent nodes
    fn get_adjacent_node_ids(&self) -> Vec<usize> {
        return self.0.adjacent_nodes.clone();
    }

    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

/// The class representing a topological edge.
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Edge")]
struct PyEdge(Edge);

#[pymethods]
impl PyEdge {

    /// Get indices of the connected nodes.
    ///
    /// :rtype: tuple[int, int]
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

/// The class representing a topology graph.
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Topology")]
struct PyTopology(Topology);

#[pymethods]
impl PyTopology {
    /// Get a list of all nodes in the topology.
    ///
    /// :rtype: list[Node]
    fn get_nodes(&self) -> Vec<PyNode> {
        return self.0.nodes_iter().map(|node| PyNode(node.clone())).collect_vec();
    }

    /// Get a list of all edges in the topology.
    ///
    /// :rtype: list[Edge]
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

/// The class representing a list of topologies.
#[pyclass]
#[pyo3(name = "TopologyContainer")]
struct PyTopologyContainer(TopologyContainer);

#[pymethods]
impl PyTopologyContainer {

    /// Query whether there is a topology in the container, which would be selected by `selector`.
    ///
    /// :param selector:
    /// :type selector: TopologySelector
    /// :return: None if no topology is selected, the position of the first selected diagram otherwise
    /// :rtype: None | int
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

/// The main generator class of the topology module.
///
/// Examples
/// --------
///
/// .. code-block:: python
///
///     model = TopologyModel([3, 4])
///     selector = TopologySelector()
///     selector.select_opi_components(1)
///     generator = TopologyGenerator(4, 3, model, selector)
///     topologies = generator.generate()
///     assert(len(topologies), 6166)
#[pyclass]
#[pyo3(name = "TopologyGenerator")]
struct PyTopologyGenerator(TopologyGenerator);

#[pymethods]
impl PyTopologyGenerator {

    /// Create a new topology generator.
    ///
    /// :param n_external: number of external legs
    /// :type n_external: int
    /// :param n_loops: number of loops
    /// :type n_loops: int
    /// :param model: the topological model used to generate the topologies
    /// :type model: TopologyModel
    /// :param selector: the selector choosing whether a given topology is kept or discarded. If no selector is specified,
    /// all topologies are kept
    /// :type selector: None | TopologySelector
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

    /// Generate the topologies for the current configuration
    ///
    /// :rtype: TopologyContainer
    fn generate(&self, py: Python<'_>) -> PyTopologyContainer {
        return py.allow_threads(
            || -> PyTopologyContainer {
                return PyTopologyContainer(self.0.generate());
            }
        );
    }
}

/// The class representing a propagator in a Feynman diagram
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Propagator")]
struct PyPropagator(Propagator);

#[pymethods]
impl PyPropagator {

    /// Get the ids of the vertices connected by this propagator
    ///
    /// :rtype: tuple[int, int]
    fn get_vertices(&self) -> (usize, usize) {
        return self.0.vertices.clone();
    }

    /// Get the momentum of the propagator, represented as the coefficients of the respective momentum. The first $N$
    /// entries are the $N$ external momenta, the remaining integrals are the loop momenta.
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///     model = Model.from_ufo("tests/QCD_UFO")
    ///     diag = DiagramGenerator(["u", "d"], ["u", "d"], 0, model).generate()[0]
    ///     # Only t-channel diagram with momentum p1 - p3
    ///     assert(diag.get_propagators()[-1].get_momentum(), [1, 0, -1, 0])
    ///
    /// :rtype: list[int]
    fn get_momentum(&self) -> Vec<i8> {
        return self.0.momentum.clone();
    }
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

/// The class representing a vertex in a Feynman diagram
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Vertex")]
struct PyVertex(Vertex);

#[pymethods]
impl PyVertex {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

/// The class representing a Feynman diagram
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Diagram")]
struct PyDiagram(Diagram);

#[pymethods]
impl PyDiagram {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.0);
    }

    fn __str__(&self) -> String {
        return format!("{}", self.0);
    }

    /// Get all propagators of the diagram, including external legs
    ///
    /// :rtype: list[Propagator]
    fn get_propagators(&self) -> Vec<PyPropagator> {
        return self.0.propagators_iter().map(|prop| PyPropagator(prop.clone())).collect_vec();
    }
}

/// A selector class which determines whether a diagram is to be kept or to be discarded. There are two types of
/// criteria available:
///
/// - opi components: select only diagrams for which the number of one-particle-irreducible components matches any of
/// the given counts
/// - custom functions: select only diagrams for which any of the given custom functions return `true`
///
#[derive(Clone)]
#[pyclass]
#[pyo3(name = "DiagramSelector")]
struct PyDiagramSelector(DiagramSelector);

#[pymethods]
impl PyDiagramSelector {

    /// Create a new topology selector, which contains no criteria and thus selects every diagram.
    #[new]
    fn new() -> Self {
        return Self(DiagramSelector::default());
    }

    /// Add a constraints to only select diagrams with `opi_count` one-particle-irreducible components.
    ///
    /// :param opi_count:
    /// :type opi_count: int
    fn select_opi_components(&mut self, opi_count: usize) {
        self.0.add_opi_count(opi_count);
    }

    /// Add a constraint to only select diagrams for which the given function returns `true`. The function receives
    /// a single diagrams as input and should return a boolean.
    ///
    /// Examples
    /// --------
    ///
    /// .. code-block:: python
    ///
    ///     def s_channel(diag: feyngraph.topology.Diagram) -> bool:
    ///         n_momenta = len(diag.get_propagators()[0].get_momentum()) # Total number of momenta in the process
    ///         s_momentum = [1, 1]+ [0]*(n_momenta-2) # e.g. = [1, 1, 0, 0] for n_momenta = 4
    ///         return any(propagator.get_momentum() == s_momentum for propagator in diag.get_propagators())
    ///
    ///     selector = feyngraph.topology.DiagramSelector()
    ///     selector.add_custom_function(s_channel)
    ///
    /// :param py_function:
    /// :type py_function: Callable[[Diagram], bool]
    fn add_custom_function(&mut self, py_function: Py<PyFunction>) {
        self.0.add_custom_function(
            Arc::new(move |diag: &Diagram| -> bool {
                Python::with_gil( |py| -> bool {
                    py_function.call1(py, (PyDiagram(diag.clone()),)).unwrap().extract(py).unwrap()
                }
                )

            })
        )
    }
}

/// The class representing a list of diagrams.
#[pyclass]
#[pyo3(name = "DiagramContainer")]
struct PyDiagramContainer(DiagramContainer);

#[pymethods]
impl PyDiagramContainer {

    /// Query whether there is a diagram in the container, which would be selected by `selector`.
    ///
    /// :param selector:
    /// :type selector: TopologySelector
    /// :return: None if no diagram is selected, the position of the first selected diagram otherwise
    /// :rtype: None | int
    fn query(&self, selector: &PyDiagramSelector) -> Option<usize> {
        return self.0.query(&selector.0);
    }

    fn __len__(&self) -> usize {
        return self.0.len();
    }
    fn __getitem__(&self, i: usize) -> PyDiagram {
        return PyDiagram((*self.0.get(i)).clone());
    }
}

/// The main class used to generate Feynman diagrams.
///
/// Examples
/// --------
///
/// .. code-block:: python
///
///     model = Model.from_ufo("tests/Standard_Model_UFO")
///     selector = DiagramSelector()
///     selector.set_opi_components(1)
///     diags = DiagramGenerator(["g", "g"], ["u", "u__tilde__", "g"], 1, model, selector).generate()
///     assert(len(diags), 51)
#[pyclass]
#[pyo3(name = "DiagramGenerator")]
struct PyDiagramGenerator(DiagramGenerator);

#[pymethods]
impl PyDiagramGenerator {

    /// Create a new Diagram generator for the given process
    ///
    /// :param incoming: list of incoming particles
    /// :type incoming: list[string]
    /// :param outgoing: list of outgoing particles
    /// :type outgoing: list[string]
    /// :param n_loops: loop order
    /// :type n_loops: int
    /// :param model:
    /// :type model: Model
    /// :param selector: selector determining which diagrams are discarded during the generation
    /// :type selector: None | DiagramSelector
    #[new]
    #[pyo3(signature = (incoming, outgoing, n_loops, model, selector=None))]
    fn new(incoming: Vec<String>,
           outgoing: Vec<String>,
           n_loops: usize,
           model: PyModel,
           selector: Option<PyDiagramSelector>
    ) -> PyResult<PyDiagramGenerator> {
        let incoming = incoming.into_iter().map(
            |particle_string| model.0.find_particle(&particle_string).cloned()
        ).try_collect()?;
        let outgoing = outgoing.into_iter().map(
            |particle_string| model.0.find_particle(&particle_string).cloned()
        ).try_collect()?;
        return if let Some(selector) = selector {
            Ok(Self(DiagramGenerator::new(incoming, outgoing, n_loops, model.0, Some(selector.0))))
        } else {
            Ok(Self(DiagramGenerator::new(incoming, outgoing, n_loops, model.0, None)))
        }
    }

    /// Generate the diagrams of the given process
    ///
    /// :rtype: DiagramContainer
    fn generate(&self, py: Python<'_>) -> PyDiagramContainer {
        return py.allow_threads(
            || -> PyDiagramContainer {
                return PyDiagramContainer(self.0.generate());
            }
        );
    }

    /// Assign particles within the current process to the topology `topo`
    ///
    /// :param topo: topology to assign particles to
    /// :type topo: Topology
    /// :rtype: DiagramContainer
    fn assign_topology(&self, py: Python<'_>, topo: &PyTopology) -> PyDiagramContainer {
        return py.allow_threads(
            || -> PyDiagramContainer {
                return PyDiagramContainer(self.0.assign_topology(&topo.0));
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