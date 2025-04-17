#![cfg(not(doctest))]

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyIndexError, PySyntaxError, PyValueError};
use std::path::PathBuf;
use std::sync::Arc;
use std::fmt::Write;
use itertools::Itertools;
use either::Either;
use pyo3::types::{PyDict, PyFunction};
use crate::{model::{
    ModelError, Model, TopologyModel, Particle, InteractionVertex
}, topology::{
    Node, Edge, Topology, TopologyContainer, TopologyGenerator,
    filter::TopologySelector
}, diagram::{
    DiagramContainer, DiagramGenerator, Diagram, filter::DiagramSelector, Propagator, Vertex, Leg,
    view::DiagramView
}, util};

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
    m.add_function(wrap_pyfunction!(generate_diagrams, m)?)?;
    return Ok(());
}

#[pyfunction]
fn set_threads(n_threads: usize) {
    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global().unwrap();
}

#[pyfunction]
#[pyo3(signature = (
    particles_in,
    particles_out,
    n_loops = 0,
    model = PyModel::__new__(),
    diagram_selector = None,
))]
fn generate_diagrams(
    py: Python<'_>,
    particles_in: Vec<String>,
    particles_out: Vec<String>,
    n_loops: usize,
    model: PyModel,
    diagram_selector: Option<PyDiagramSelector>,
) -> PyResult<PyDiagramContainer> {
    let mut selector;
    if let Some(in_selector) = diagram_selector {
        selector = in_selector;
    } else {
        selector = PyDiagramSelector::new();
        if n_loops > 0 {
            selector.select_opi_components(1);
        }
    }
    return Ok(PyDiagramGenerator::new(particles_in, particles_out, n_loops, model, Some(selector))?.generate(py));
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

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Particle")]
struct PyParticle(Particle);

#[pymethods]
impl PyParticle {
    fn name(&self) -> String { return self.0.get_name().clone(); }

    fn anti_name(&self) -> String { return self.0.get_anti_name().clone(); }

    fn is_anti(&self) -> bool { return self.0.is_anti(); }

    fn is_fermi(&self) -> bool { return self.0.is_fermi() }

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
struct PyInteractionVertex(InteractionVertex);

#[pymethods]
impl PyInteractionVertex {
    fn __repr__(&self) -> String { return format!("{:#?}", self.0); }

    fn __str__(&self) -> String { return format!("{:?}", self.0); }

    fn coupling_orders(&self) -> HashMap<String, usize> {
        return self.0.coupling_orders.clone();
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

    fn select_node_degree_range(&mut self, degree: usize, start: usize, end: usize) {
        self.0.add_node_degree_range(degree, start..end);
    }

    fn select_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.0.add_node_partition(partition);
    }

    fn select_opi_components(&mut self, opi_count: usize) {
        self.0.add_opi_count(opi_count);
    }

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
struct PyTopology(Topology);

#[pymethods]
impl PyTopology {
    fn nodes(&self) -> Vec<PyNode> {
        return self.0.nodes_iter().map(|node| PyNode(node.clone())).collect_vec();
    }

    fn edges(&self) -> Vec<PyEdge> {
        return self.0.edges_iter().map(|edge| PyEdge(edge.clone())).collect_vec();
    }

    fn symmetry_factor(&self) -> usize { return self.0.node_symmetry * self.0.edge_symmetry; }

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
    fn __getitem__(&self, i: usize) -> PyResult<PyTopology> {
        if i >= self.0.len() {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        return Ok(PyTopology((*self.0.get(i)).clone()));
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

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Leg")]
struct PyLeg {
    container: Arc<DiagramContainer>,
    diagram: Arc<PyDiagram>,
    leg: Arc<Leg>,
    leg_index: usize,
    invert_particle: bool
}

#[pymethods]
impl PyLeg {
    pub fn vertices(&self) -> Vec<PyVertex> {
        return vec![PyVertex {
            container: self.container.clone(),
            diagram: self.diagram.clone(),
            vertex: Arc::new(self.diagram.diagram.vertices[self.leg.vertex].clone()),
            index: self.leg.vertex
        }];
    }
    pub fn vertex(&self, _index: usize) -> PyVertex {
        return PyVertex {
            container: self.container.clone(),
            diagram: self.diagram.clone(),
            vertex: Arc::new(self.diagram.diagram.vertices[self.leg.vertex].clone()),
            index: self.leg.vertex
        };
    }

    pub fn particle(&self) -> PyParticle {
        return if self.invert_particle {
            PyParticle(self.container.model.as_ref().unwrap().get_anti(self.leg.particle).clone())
        } else {
            PyParticle(self.container.model.as_ref().unwrap().get_particle(self.leg.particle).clone())
        }
    }

    pub fn ray_index(&self, _vertex: usize) -> usize {
        return self.diagram.diagram.vertices[self.leg.vertex]
            .propagators.iter().position(
            |p| (*p + self.diagram.n_ext() as isize) as usize == self.leg_index
        ).unwrap();
    }

    pub fn momentum(&self) -> Vec<i8> {
        return self.leg.momentum.clone();
    }

    pub fn momentum_str(&self) -> String {
        let momentum_labels = &self.container.momentum_labels;
        let mut result = String::with_capacity(5*momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.leg.momentum.iter().enumerate() {
            if *coefficient == 0 { continue; }
            let sign;
            if first {
                sign = "";
                first = false;
            } else {
                sign = "+";
            }
            match *coefficient {
                1 => write!(&mut result, "{}{}", sign, momentum_labels[i]).unwrap(),
                -1 => write!(&mut result, "-{}", momentum_labels[i]).unwrap(),
                x if x < 0 => write!(&mut result, "-{}*{}", x.abs(), momentum_labels[i]).unwrap(),
                x => write!(&mut result, "{}{}*{}", sign, x, momentum_labels[i]).unwrap()
            }
        }
        return result;
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Propagator")]
struct PyPropagator {
    container: Arc<DiagramContainer>,
    diagram: Arc<PyDiagram>,
    propagator: Arc<Propagator>,
    index: usize,
    invert: bool
}

#[pymethods]
impl PyPropagator {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.propagator);
    }

    fn __str__(&self) -> String {
        return format!("{}", self);
    }

    pub fn vertices(&self) -> Vec<PyVertex> {
        return if self.invert {
            self.propagator.vertices.iter().rev().map(
                |i| PyVertex {
                    container: self.container.clone(),
                    diagram: self.diagram.clone(),
                    vertex: Arc::new(self.diagram.diagram.vertices[*i].clone()),
                    index: *i
                }
            ).collect_vec()
        } else {
            self.propagator.vertices.iter().map(
                |i| PyVertex {
                    container: self.container.clone(),
                    diagram: self.diagram.clone(),
                    vertex: Arc::new(self.diagram.diagram.vertices[*i].clone()),
                    index: *i
                }
            ).collect_vec()
        }
    }

    pub fn vertex(&self, index: usize) -> PyVertex {
        let i = if self.invert {
            1 - index
        } else {
            index
        };
        return PyVertex {
            container: self.container.clone(),
            diagram: self.diagram.clone(),
            vertex: Arc::new(self.diagram.diagram.vertices[
                self.propagator.vertices[i]
                ].clone()),
            index: self.propagator.vertices[i]
        };
    }

    pub fn particle(&self) -> PyParticle {
        return if self.invert {
            PyParticle(self.container.model.as_ref().unwrap().get_anti(self.propagator.particle).clone())
        } else {
            PyParticle(self.container.model.as_ref().unwrap().get_particle(self.propagator.particle).clone())
        }
    }

    pub fn ray_index(&self, index: usize) -> usize {
        let i = if self.invert {
            1 - index
        } else {
            index
        };
        return self.diagram.diagram.vertices[self.propagator.vertices[i]]
            .propagators.iter().position(|p| *p == self.index as isize).unwrap();
    }

    pub fn momentum(&self) -> Vec<i8> {
        return if self.invert {
            self.propagator.momentum.iter().map(|x| -*x).collect_vec()
        } else {
            self.propagator.momentum.clone()
        }
    }

    pub fn momentum_str(&self) -> String {
        let momentum_labels = &self.container.momentum_labels;
        let mut result = String::with_capacity(5*momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.propagator.momentum.iter().enumerate() {
            if *coefficient == 0 { continue; }
            let sign;
            if first {
                sign = "";
                first = false;
            } else {
                sign = "+";
            }
            match *coefficient * if self.invert {-1} else {1} {
                1 => write!(&mut result, "{}{}", sign, momentum_labels[i]).unwrap(),
                -1 => write!(&mut result, "-{}", momentum_labels[i]).unwrap(),
                x if x < 0 => write!(&mut result, "-{}*{}", x.abs(), momentum_labels[i]).unwrap(),
                x => write!(&mut result, "{}{}*{}", sign, x, momentum_labels[i]).unwrap()
            }
        }
        return result;
    }

    pub fn id(&self) -> usize { return self.index; }
}

impl std::fmt::Display for PyPropagator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[{} -> {}], p = {},",
               self.particle().name(),
               self.propagator.vertices[0],
               self.propagator.vertices[1],
               self.momentum_str()
        )?;
        Ok(())
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Vertex")]
struct PyVertex {
    container: Arc<DiagramContainer>,
    diagram: Arc<PyDiagram>,
    vertex: Arc<Vertex>,
    index: usize,
}

#[pymethods]
impl PyVertex {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.vertex);
    }

    fn __str__(&self) -> String {
        return format!("{}", self);
    }

    pub fn propagators(&self) -> Vec<Either<PyLeg, PyPropagator>> {
        return self.vertex.propagators.iter().map(
            |i| if *i >= 0 {
                Either::Right(PyPropagator {
                    container: self.container.clone(),
                    diagram: self.diagram.clone(),
                    propagator: Arc::new(self.diagram.diagram.propagators[*i as usize].clone()),
                    index: *i as usize,
                    invert: self.diagram.diagram.propagators[*i as usize].vertices[0] == self.index
                })
            } else {
                let index = (*i + self.diagram.n_ext() as isize) as usize;
                let leg = if index < self.diagram.diagram.incoming_legs.len() {
                    &self.diagram.diagram.incoming_legs[index]
                } else {
                    &self.diagram.diagram.outgoing_legs[index - self.diagram.diagram.incoming_legs.len()]
                };
                Either::Left(PyLeg {
                    container: self.container.clone(),
                    diagram: self.diagram.clone(),
                    leg: Arc::new(leg.clone()),
                    leg_index: index,
                    invert_particle: false,
                })
            }
        ).collect_vec();
    }

    pub fn propagators_ordered(&self) -> Vec<Either<PyLeg, PyPropagator>> {
        let props= self.propagators();
        let mut perm = Vec::with_capacity(self.vertex.propagators.len());
        let mut seen = vec![false; self.vertex.propagators.len()];
        for ref_particle in self.container.model.as_ref().unwrap().vertex(self.vertex.interaction).particles.iter() {
            for (i, part) in props.iter().map(
                |prop| either::for_both!(prop, p => p.particle())
            ).enumerate() {
                if !seen[i] && part.name() == *ref_particle {
                    perm.push(i);
                    seen[i] = true;
                } else {
                    continue;
                }
            }
        }
        return perm.into_iter().map(|i| props[i].clone()).collect_vec();
    }

    pub fn interaction(&self) -> PyInteractionVertex {
        return PyInteractionVertex(self.container.model.as_ref().unwrap().vertex(self.vertex.interaction).clone());
    }

    pub fn particles_ordered(&self) -> Vec<PyParticle> {
        return self.interaction().0.particles.iter().map(
            |p| PyParticle(self.container.model.as_ref().unwrap().get_particle_name(p).unwrap().clone())
        ).collect_vec();
    }

    pub fn match_particles(&self, query: Vec<String>) -> bool {
        return self.container.model.as_ref().unwrap().vertex(self.vertex.interaction).particles.iter().sorted()
            .zip(query.into_iter().sorted()).all(
            |(part, query)| *part == *query
        );
    }

    pub fn id(&self) -> usize { return self.index; }

    pub fn degree(&self) -> usize { return self.vertex.propagators.len(); }
}

impl std::fmt::Display for PyVertex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[ ", self.container.model.as_ref().unwrap().vertex(self.vertex.interaction).name)?;
        for p in self.container.model.as_ref().unwrap().vertex(self.vertex.interaction).particles.iter() {
            write!(f, "{} ", p)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Diagram")]
struct PyDiagram {
    pub(crate) diagram: Arc<Diagram>,
    pub(crate) container: Arc<DiagramContainer>
}

#[pymethods]
impl PyDiagram {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.diagram);
    }

    fn __str__(&self) -> String {
        return format!("{}", DiagramView::new(
                self.container.model.as_ref().unwrap(),
                self.diagram.as_ref(),
                &self.container.momentum_labels
            ));
    }

    pub fn incoming(&self) -> Vec<PyLeg> {
        return self.diagram.incoming_legs.iter().enumerate().map(
            |(i, p)| PyLeg {
                container: self.container.clone(),
                diagram: Arc::new(self.clone()),
                leg: Arc::new(p.clone()),
                leg_index: i,
                invert_particle: false
            }
        ).collect_vec();
    }

    pub fn outgoing(&self) -> Vec<PyLeg> {
        return self.diagram.outgoing_legs.iter().enumerate().map(
            |(i, p)| PyLeg {
                container: self.container.clone(),
                diagram: Arc::new(self.clone()),
                leg: Arc::new(p.clone()),
                leg_index: i + self.diagram.incoming_legs.len(),
                invert_particle: true
            }
        ).collect_vec();
    }

    pub fn propagators(&self) -> Vec<PyPropagator> {
        return self.diagram.propagators.iter().enumerate()
            .map(
                |(i, p)| PyPropagator {
                    container: self.container.clone(),
                    diagram: Arc::new(self.clone()),
                    propagator: Arc::new(p.clone()),
                    index: i,
                    invert: false
                }
            ).collect_vec();
    }

    pub fn propagator(&self, index: usize) -> PyPropagator {
        return PyPropagator {
            container: self.container.clone(),
            diagram: Arc::new(self.clone()),
            propagator: Arc::new(self.diagram.propagators[index].clone()),
            index,
            invert: false
        }
    }

    pub fn vertex(&self, index: usize) -> PyVertex {
        return PyVertex {
            container: self.container.clone(),
            diagram: Arc::new(self.clone()),
            vertex: Arc::new(self.diagram.vertices[index].clone()),
            index
        }
    }

    pub fn vertices(&self) -> Vec<PyVertex> {
        return self.diagram.vertices.iter().enumerate().map(
            |(i, v)| PyVertex {
                container: self.container.clone(),
                diagram: Arc::new(self.clone()),
                vertex: Arc::new(v.clone()),
                index: i
            }
        ).collect_vec();
    }

    pub fn loop_vertices(&self, index: usize) -> Vec<PyVertex> {
        let loop_index = self.n_ext() + index;
        return self.diagram.vertices.iter().enumerate().filter_map(
            |(i, v)| if self.diagram.vertices[index].propagators.iter().any(
                |j| *j >= 0 && self.diagram.propagators[*j as usize].momentum[loop_index] != 0
            ) {
                Some(PyVertex {
                    container: self.container.clone(),
                    diagram: Arc::new(self.clone()),
                    vertex: Arc::new(v.clone()),
                    index: i
                })
            } else {
                None
            }
        ).collect_vec();
    }

    pub fn chord(&self, index: usize) -> Vec<PyPropagator> {
        let loop_index = self.n_ext() + index;
        return self.diagram.propagators.iter().enumerate().filter_map(
            |(i, prop)| if prop.momentum[loop_index] != 0 {
                Some(self.propagator(i))
            } else {
                None
            }
        ).collect_vec();
    }

    pub fn bridges(&self) -> Vec<PyPropagator> {
        return self.diagram.bridges.iter().map(
            |i| self.propagator(*i)
        ).collect_vec();
    }

    pub fn n_ext(&self) -> usize {
        return self.diagram.incoming_legs.len() + self.diagram.outgoing_legs.len();
    }

    pub fn symmetry_factor(&self) -> usize {
        return self.diagram.vertex_symmetry * self.diagram.propagator_symmetry;
    }

    pub fn sign(&self) -> i8 {
        return self.diagram.sign;
    }

}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "DiagramSelector")]
struct PyDiagramSelector(DiagramSelector);

#[pymethods]
impl PyDiagramSelector {
    #[new]
    fn new() -> Self {
        return Self(DiagramSelector::default());
    }

    fn select_opi_components(&mut self, opi_count: usize) {
        self.0.add_opi_count(opi_count);
    }

    fn select_self_loops(&mut self, count: usize) { self.0.add_self_loop_count(count); }

    fn select_on_shell(&mut self) { self.0.set_on_shell(); }

    fn add_custom_function(&mut self, py_function: Py<PyFunction>) {
        self.0.add_unwrapped_custom_function(
            Arc::new(move |model: Arc<Model>, momentum_labels: Arc<Vec<String>>, diag: &Diagram| -> bool {
                let py_diag = PyDiagram {
                    diagram: Arc::new(diag.clone()),
                    container: Arc::new(DiagramContainer {
                        model: Some(model),
                        momentum_labels,
                        data: vec![]
                    })
                };
                Python::with_gil( |py| -> bool {
                    py_function.call1(py, (py_diag,)).unwrap().extract(py).unwrap()
                }
                )

            })
        )
    }

    fn add_coupling_power(&mut self, coupling: String, power: usize) {
        self.0.add_coupling_power(&coupling, power);
    }

    fn add_propagator_count(&mut self, particle: String, count: usize) {
        self.0.add_propagator_count(&particle, count);
    }

    fn add_vertex_count(&mut self, particles: Vec<String>, count: usize) {
        self.0.add_vertex_count(particles, count);
    }

    fn __deepcopy__(&self, _memo: Py<PyDict>) -> Self { return self.clone(); }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "DiagramContainer")]
struct PyDiagramContainer(Arc<DiagramContainer>);

#[pymethods]
impl PyDiagramContainer {

    fn query(&self, selector: &PyDiagramSelector) -> Option<usize> {
        return self.0.query(&selector.0);
    }

    fn __len__(&self) -> usize {
        return self.0.len();
    }
    fn __getitem__(&self, i: usize) -> PyResult<PyDiagram> {
        if i >= self.0.len() {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        return Ok(PyDiagram {
            container: self.0.clone(),
            diagram: Arc::new(self.0.data[i].clone()),
        });
    }
}

#[pyclass]
#[pyo3(name = "DiagramGenerator")]
struct PyDiagramGenerator(DiagramGenerator);

#[pymethods]
impl PyDiagramGenerator {

    #[new]
    #[pyo3(signature = (incoming, outgoing, n_loops, model, selector=None))]
    fn new(incoming: Vec<String>,
           outgoing: Vec<String>,
           n_loops: usize,
           model: PyModel,
           selector: Option<PyDiagramSelector>
    ) -> PyResult<PyDiagramGenerator> {
        let incoming = incoming.into_iter().map(
            |particle_string| model.0.get_particle_index(&particle_string)
        ).try_collect()?;
        let outgoing = outgoing.into_iter().map(
            |particle_string| model.0.get_particle_index(&particle_string)
        ).try_collect()?;
        return if let Some(selector) = selector {
            Ok(Self(DiagramGenerator::new(incoming, outgoing, n_loops, model.0, Some(selector.0))))
        } else {
            Ok(Self(DiagramGenerator::new(incoming, outgoing, n_loops, model.0, None)))
        }
    }

    fn set_momentum_labels(&mut self, labels: Vec<String>) -> PyResult<()> {
        self.0.set_momentum_labels(labels)?;
        return Ok(());
    }

    fn generate(&self, py: Python<'_>) -> PyDiagramContainer {
        return py.allow_threads(
            || -> PyDiagramContainer {
                return PyDiagramContainer(Arc::new(self.0.generate()));
            }
        );
    }

    fn assign_topology(&self, py: Python<'_>, topo: &PyTopology) -> PyDiagramContainer {
        return py.allow_threads(
            || -> PyDiagramContainer {
                return PyDiagramContainer(Arc::new(self.0.assign_topology(&topo.0)));
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
    for edge in topo.edges():
        nodes = edge.nodes()
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