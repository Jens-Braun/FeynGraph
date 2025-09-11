use super::{PyInteractionVertex, PyModel, PyParticle, topology::PyTopology};
use crate::{
    diagram::{
        Diagram, DiagramContainer, DiagramGenerator, Leg, Propagator, Vertex, filter::DiagramSelector,
        view::DiagramView,
    },
    model::Model,
    topology::Topology,
};
use either::Either;
use itertools::Itertools;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Leg")]
pub(crate) struct PyLeg {
    container: Arc<DiagramContainer>,
    diagram: Arc<PyDiagram>,
    leg: Arc<Leg>,
    leg_index: usize,
    invert_particle: bool,
}

#[pymethods]
impl PyLeg {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.leg);
    }

    fn __str__(&self) -> String {
        return format!("{}", self);
    }

    pub fn vertices(&self) -> Vec<PyVertex> {
        return vec![PyVertex {
            container: self.container.clone(),
            diagram: self.diagram.clone(),
            vertex: Arc::new(self.diagram.diagram.vertices[self.leg.vertex].clone()),
            index: self.leg.vertex,
        }];
    }

    #[pyo3(signature = (_index = 0))]
    pub fn vertex(&self, _index: usize) -> PyVertex {
        return PyVertex {
            container: self.container.clone(),
            diagram: self.diagram.clone(),
            vertex: Arc::new(self.diagram.diagram.vertices[self.leg.vertex].clone()),
            index: self.leg.vertex,
        };
    }

    pub fn particle(&self) -> PyParticle {
        return if self.invert_particle {
            PyParticle(
                self.container
                    .model
                    .as_ref()
                    .unwrap()
                    .get_anti(self.leg.particle)
                    .clone(),
            )
        } else {
            PyParticle(
                self.container
                    .model
                    .as_ref()
                    .unwrap()
                    .get_particle(self.leg.particle)
                    .clone(),
            )
        };
    }

    #[pyo3(signature = (_vertex = 0))]
    pub fn ray_index(&self, _vertex: usize) -> usize {
        return self.diagram.diagram.vertices[self.leg.vertex]
            .propagators
            .iter()
            .position(|p| (*p + self.diagram.n_ext() as isize) as usize == self.leg_index)
            .unwrap();
    }

    pub fn id(&self) -> usize {
        return self.leg_index;
    }

    pub fn momentum(&self) -> Vec<i8> {
        return self.leg.momentum.clone();
    }

    pub fn momentum_str(&self) -> String {
        let momentum_labels = &self.container.momentum_labels;
        let mut result = String::with_capacity(5 * momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.leg.momentum.iter().enumerate() {
            if *coefficient == 0 {
                continue;
            }
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
                x => write!(&mut result, "{}{}*{}", sign, x, momentum_labels[i]).unwrap(),
            }
        }
        return result;
    }
}

impl std::fmt::Display for PyLeg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.leg_index >= self.diagram.diagram.incoming_legs.len() {
            write!(
                f,
                "{}[{} -> ], p = {},",
                self.particle().name(),
                self.leg.vertex,
                self.momentum_str()
            )?;
        } else {
            write!(
                f,
                "{}[-> {}], p = {},",
                self.particle().name(),
                self.leg.vertex,
                self.momentum_str()
            )?;
        }
        Ok(())
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Propagator")]
pub(crate) struct PyPropagator {
    container: Arc<DiagramContainer>,
    diagram: Arc<PyDiagram>,
    propagator: Arc<Propagator>,
    index: usize,
    invert: bool,
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
            self.propagator
                .vertices
                .iter()
                .rev()
                .map(|i| PyVertex {
                    container: self.container.clone(),
                    diagram: self.diagram.clone(),
                    vertex: Arc::new(self.diagram.diagram.vertices[*i].clone()),
                    index: *i,
                })
                .collect_vec()
        } else {
            self.propagator
                .vertices
                .iter()
                .map(|i| PyVertex {
                    container: self.container.clone(),
                    diagram: self.diagram.clone(),
                    vertex: Arc::new(self.diagram.diagram.vertices[*i].clone()),
                    index: *i,
                })
                .collect_vec()
        };
    }

    pub fn vertex(&self, index: usize) -> PyVertex {
        let i = if self.invert { 1 - index } else { index };
        return PyVertex {
            container: self.container.clone(),
            diagram: self.diagram.clone(),
            vertex: Arc::new(self.diagram.diagram.vertices[self.propagator.vertices[i]].clone()),
            index: self.propagator.vertices[i],
        };
    }

    pub fn particle(&self) -> PyParticle {
        return if self.invert {
            PyParticle(
                self.container
                    .model
                    .as_ref()
                    .unwrap()
                    .get_anti(self.propagator.particle)
                    .clone(),
            )
        } else {
            PyParticle(
                self.container
                    .model
                    .as_ref()
                    .unwrap()
                    .get_particle(self.propagator.particle)
                    .clone(),
            )
        };
    }

    pub fn ray_index(&self, index: usize) -> usize {
        let i = if self.invert { 1 - index } else { index };
        let pos = self.diagram.diagram.vertices[self.propagator.vertices[i]]
            .propagators
            .iter()
            .position(|p| *p == self.index as isize)
            .unwrap();
        return if self.propagator.vertices[0] == self.propagator.vertices[1] {
            // If the propagator is a self-loop, the vertex' propagator list contains `self.index` twice - at `pos` and at `pos+1`.
            // Since the direction of a self-loop is always ambiguous, it is fixed such that it always runs from the leg at `pos`
            // to the leg at `pos+1`.
            pos + i
        } else {
            pos
        };
    }

    pub fn ray_index_ordered(&self, index: usize) -> usize {
        let i = if self.invert { 1 - index } else { index };
        return self
            .vertex(i)
            .propagators_ordered()
            .iter()
            .position(|p| {
                if let either::Right(p) = p {
                    if p.index == self.index {
                        if self.propagator.vertices[0] == self.propagator.vertices[1] {
                            if i == 1 {
                                return p.particle().is_anti();
                            } else {
                                return !p.particle().is_anti();
                            }
                        } else {
                            return true;
                        }
                    } else {
                        return false;
                    }
                } else {
                    false
                }
            })
            .unwrap();
    }

    pub fn momentum(&self) -> Vec<i8> {
        return if self.invert {
            self.propagator.momentum.iter().map(|x| -*x).collect_vec()
        } else {
            self.propagator.momentum.clone()
        };
    }

    pub fn momentum_str(&self) -> String {
        let momentum_labels = &self.container.momentum_labels;
        let mut result = String::with_capacity(5 * momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.propagator.momentum.iter().enumerate() {
            if *coefficient == 0 {
                continue;
            }
            let sign;
            if first {
                sign = "";
                first = false;
            } else {
                sign = "+";
            }
            match *coefficient * if self.invert { -1 } else { 1 } {
                1 => write!(&mut result, "{}{}", sign, momentum_labels[i]).unwrap(),
                -1 => write!(&mut result, "-{}", momentum_labels[i]).unwrap(),
                x if x < 0 => write!(&mut result, "-{}*{}", x.abs(), momentum_labels[i]).unwrap(),
                x => write!(&mut result, "{}{}*{}", sign, x, momentum_labels[i]).unwrap(),
            }
        }
        return result;
    }

    pub fn id(&self) -> usize {
        return self.index + self.diagram.n_ext();
    }
}

impl std::fmt::Display for PyPropagator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.invert {
            write!(
                f,
                "{}[{} -> {}], p = {},",
                self.particle().name(),
                self.propagator.vertices[1],
                self.propagator.vertices[0],
                self.momentum_str()
            )?;
        } else {
            write!(
                f,
                "{}[{} -> {}], p = {},",
                self.particle().name(),
                self.propagator.vertices[0],
                self.propagator.vertices[1],
                self.momentum_str()
            )?;
        }
        Ok(())
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Vertex")]
pub(crate) struct PyVertex {
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
        return self
            .vertex
            .propagators
            .iter()
            .enumerate()
            .map(|(i, prop)| {
                if *prop >= 0 {
                    Either::Right(PyPropagator {
                        container: self.container.clone(),
                        diagram: self.diagram.clone(),
                        propagator: Arc::new(self.diagram.diagram.propagators[*prop as usize].clone()),
                        index: *prop as usize,
                        invert: (self.diagram.diagram.propagators[*prop as usize].vertices[0] == self.index
                            && self.diagram.diagram.propagators[*prop as usize].vertices[1] != self.index)
                            || (i > 0 && self.vertex.propagators[i - 1] == self.vertex.propagators[i]),
                    })
                } else {
                    let index = (*prop + self.diagram.n_ext() as isize) as usize;
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
            })
            .collect_vec();
    }

    pub fn propagators_ordered(&self) -> Vec<Either<PyLeg, PyPropagator>> {
        let props = self.propagators();
        let mut perm = Vec::with_capacity(self.vertex.propagators.len());
        let mut seen = vec![false; self.vertex.propagators.len()];
        for ref_particle in self
            .container
            .model
            .as_ref()
            .unwrap()
            .vertex(self.vertex.interaction)
            .particles
            .iter()
        {
            for (i, part) in props
                .iter()
                .map(|prop| either::for_both!(prop, p => p.particle()))
                .enumerate()
            {
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
        return PyInteractionVertex(
            self.container
                .model
                .as_ref()
                .unwrap()
                .vertex(self.vertex.interaction)
                .clone(),
        );
    }

    pub fn particles_ordered(&self) -> Vec<PyParticle> {
        return self
            .interaction()
            .0
            .particles
            .iter()
            .map(|p| {
                PyParticle(
                    self.container
                        .model
                        .as_ref()
                        .unwrap()
                        .get_particle_by_name(p)
                        .unwrap()
                        .clone(),
                )
            })
            .collect_vec();
    }

    pub fn match_particles(&self, query: Vec<String>) -> bool {
        return self
            .container
            .model
            .as_ref()
            .unwrap()
            .vertex(self.vertex.interaction)
            .particles
            .iter()
            .sorted()
            .zip(query.into_iter().sorted())
            .all(|(part, query)| *part == *query);
    }

    pub fn id(&self) -> usize {
        return self.index;
    }

    pub fn degree(&self) -> usize {
        return self.vertex.propagators.len();
    }
}

impl std::fmt::Display for PyVertex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}[ ",
            self.container
                .model
                .as_ref()
                .unwrap()
                .vertex(self.vertex.interaction)
                .name
        )?;
        for p in self
            .container
            .model
            .as_ref()
            .unwrap()
            .vertex(self.vertex.interaction)
            .particles
            .iter()
        {
            write!(f, "{} ", p)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Diagram")]
pub(crate) struct PyDiagram {
    pub(crate) diagram: Arc<Diagram>,
    pub(crate) container: Arc<DiagramContainer>,
}

#[pymethods]
impl PyDiagram {
    fn __repr__(&self) -> String {
        return format!("{:#?}", self.diagram);
    }

    fn __str__(&self) -> String {
        return format!(
            "{}",
            DiagramView::new(
                self.container.model.as_ref().unwrap(),
                self.diagram.as_ref(),
                &self.container.momentum_labels
            )
        );
    }

    fn _repr_svg_(&self) -> String {
        return DiagramView::new(
            self.container.model.as_ref().unwrap(),
            self.diagram.as_ref(),
            &self.container.momentum_labels,
        )
        .draw_svg_str();
    }

    pub(crate) fn draw_tikz(&self, file: PathBuf) -> PyResult<()> {
        DiagramView::new(
            self.container.model.as_ref().unwrap(),
            self.diagram.as_ref(),
            &self.container.momentum_labels,
        )
        .draw_tikz(file)?;
        Ok(())
    }

    pub(crate) fn draw_svg(&self, file: PathBuf) -> PyResult<()> {
        DiagramView::new(
            self.container.model.as_ref().unwrap(),
            self.diagram.as_ref(),
            &self.container.momentum_labels,
        )
        .draw_svg(file)?;
        Ok(())
    }

    pub(crate) fn incoming(&self) -> Vec<PyLeg> {
        return self
            .diagram
            .incoming_legs
            .iter()
            .enumerate()
            .map(|(i, p)| PyLeg {
                container: self.container.clone(),
                diagram: Arc::new(self.clone()),
                leg: Arc::new(p.clone()),
                leg_index: i,
                invert_particle: false,
            })
            .collect_vec();
    }

    pub(crate) fn outgoing(&self) -> Vec<PyLeg> {
        return self
            .diagram
            .outgoing_legs
            .iter()
            .enumerate()
            .map(|(i, p)| PyLeg {
                container: self.container.clone(),
                diagram: Arc::new(self.clone()),
                leg: Arc::new(p.clone()),
                leg_index: i + self.diagram.incoming_legs.len(),
                invert_particle: true,
            })
            .collect_vec();
    }

    pub(crate) fn propagators(&self) -> Vec<PyPropagator> {
        return self
            .diagram
            .propagators
            .iter()
            .enumerate()
            .map(|(i, p)| PyPropagator {
                container: self.container.clone(),
                diagram: Arc::new(self.clone()),
                propagator: Arc::new(p.clone()),
                index: i,
                invert: false,
            })
            .collect_vec();
    }

    pub(crate) fn propagator(&self, index: usize) -> PyPropagator {
        return PyPropagator {
            container: self.container.clone(),
            diagram: Arc::new(self.clone()),
            propagator: Arc::new(self.diagram.propagators[index].clone()),
            index,
            invert: false,
        };
    }

    pub(crate) fn vertex(&self, index: usize) -> PyVertex {
        return PyVertex {
            container: self.container.clone(),
            diagram: Arc::new(self.clone()),
            vertex: Arc::new(self.diagram.vertices[index].clone()),
            index,
        };
    }

    pub(crate) fn vertices(&self) -> Vec<PyVertex> {
        return self
            .diagram
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| PyVertex {
                container: self.container.clone(),
                diagram: Arc::new(self.clone()),
                vertex: Arc::new(v.clone()),
                index: i,
            })
            .collect_vec();
    }

    pub(crate) fn loop_vertices(&self, index: usize) -> Vec<PyVertex> {
        let loop_index = self.n_ext() + index;
        return self
            .diagram
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if self.diagram.vertices[index]
                    .propagators
                    .iter()
                    .any(|j| *j >= 0 && self.diagram.propagators[*j as usize].momentum[loop_index] != 0)
                {
                    Some(PyVertex {
                        container: self.container.clone(),
                        diagram: Arc::new(self.clone()),
                        vertex: Arc::new(v.clone()),
                        index: i,
                    })
                } else {
                    None
                }
            })
            .collect_vec();
    }

    pub(crate) fn chord(&self, index: usize) -> Vec<PyPropagator> {
        let loop_index = self.n_ext() + index;
        return self
            .diagram
            .propagators
            .iter()
            .enumerate()
            .filter_map(|(i, prop)| {
                if prop.momentum[loop_index] != 0 {
                    Some(self.propagator(i))
                } else {
                    None
                }
            })
            .collect_vec();
    }

    pub(crate) fn bridges(&self) -> Vec<PyPropagator> {
        return self.diagram.bridges.iter().map(|i| self.propagator(*i)).collect_vec();
    }

    pub(crate) fn n_ext(&self) -> usize {
        return self.diagram.incoming_legs.len() + self.diagram.outgoing_legs.len();
    }

    pub(crate) fn symmetry_factor(&self) -> usize {
        return self.diagram.vertex_symmetry * self.diagram.propagator_symmetry;
    }

    pub(crate) fn sign(&self) -> i8 {
        return self.diagram.sign;
    }

    pub(crate) fn n_in(&self) -> usize {
        return self.diagram.n_in();
    }

    pub(crate) fn n_out(&self) -> usize {
        return self.diagram.n_out();
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "DiagramSelector")]
pub(crate) struct PyDiagramSelector(DiagramSelector);

#[pymethods]
impl PyDiagramSelector {
    #[new]
    pub(crate) fn new() -> Self {
        return Self(DiagramSelector::default());
    }

    pub(crate) fn select_opi_components(&mut self, opi_count: usize) {
        self.0.select_opi_components(opi_count);
    }

    fn select_self_loops(&mut self, count: usize) {
        self.0.select_self_loops(count);
    }

    fn select_on_shell(&mut self) {
        self.0.select_on_shell();
    }

    fn add_custom_function(&mut self, py_function: Py<PyAny>) {
        self.0.add_unwrapped_custom_function(Arc::new(
            move |model: Arc<Model>, momentum_labels: Arc<Vec<String>>, diag: &Diagram| -> bool {
                let py_diag = PyDiagram {
                    diagram: Arc::new(diag.clone()),
                    container: Arc::new(DiagramContainer {
                        model: Some(model),
                        momentum_labels,
                        data: vec![],
                    }),
                };
                Python::with_gil(|py| -> bool { py_function.call1(py, (py_diag,)).unwrap().extract(py).unwrap() })
            },
        ))
    }

    fn add_topology_function(&mut self, py_function: Py<PyAny>) {
        self.0.add_topology_function(Arc::new(move |topo: &Topology| -> bool {
            Python::with_gil(|py| -> bool {
                py_function
                    .call1(py, (PyTopology(topo.clone()),))
                    .unwrap()
                    .extract(py)
                    .unwrap()
            })
        }))
    }

    fn select_coupling_power(&mut self, coupling: String, power: usize) {
        self.0.select_coupling_power(&coupling, power);
    }

    fn select_propagator_count(&mut self, particle: String, count: usize) {
        self.0.select_propagator_count(&particle, count);
    }

    fn select_vertex_count(&mut self, particles: Vec<String>, count: usize) {
        self.0.select_vertex_count(particles, count);
    }

    fn select_vertex_degree(&mut self, degree: usize, count: usize) {
        self.0.select_vertex_degree(degree, count);
    }

    fn __deepcopy__(&self, _memo: Py<PyDict>) -> Self {
        return self.clone();
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "DiagramContainer")]
pub(crate) struct PyDiagramContainer(Arc<DiagramContainer>);

#[pymethods]
impl PyDiagramContainer {
    fn query(&self, selector: &PyDiagramSelector) -> Option<usize> {
        return self.0.query(&selector.0);
    }

    #[pyo3(signature = (diagrams, n_cols = None))]
    fn draw(&self, diagrams: Vec<usize>, n_cols: Option<usize>) -> String {
        return self.0.draw_svg(&diagrams, n_cols);
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

    fn _repr_svg_(&self) -> String {
        let n = self.0.len().min(100);
        return self.0.draw_svg(&(0..n).collect_vec(), None);
    }
}

#[pyclass]
#[pyo3(name = "DiagramGenerator")]
pub(crate) struct PyDiagramGenerator(DiagramGenerator);

#[pymethods]
impl PyDiagramGenerator {
    #[new]
    #[pyo3(signature = (incoming, outgoing, n_loops, model, selector=None))]
    pub(crate) fn new(
        incoming: Vec<String>,
        outgoing: Vec<String>,
        n_loops: usize,
        model: PyModel,
        selector: Option<PyDiagramSelector>,
    ) -> PyResult<PyDiagramGenerator> {
        let incoming = incoming.iter().map(String::as_ref).collect_vec();
        let outgoing = outgoing.iter().map(String::as_ref).collect_vec();
        return if let Some(selector) = selector {
            Ok(Self(DiagramGenerator::new(
                &incoming,
                &outgoing,
                n_loops,
                model.0,
                Some(selector.0),
            )?))
        } else {
            Ok(Self(DiagramGenerator::new(
                &incoming, &outgoing, n_loops, model.0, None,
            )?))
        };
    }

    fn set_momentum_labels(&mut self, labels: Vec<String>) -> PyResult<()> {
        self.0.set_momentum_labels(labels)?;
        return Ok(());
    }

    pub(crate) fn generate(&self, py: Python<'_>) -> PyDiagramContainer {
        return py.allow_threads(|| -> PyDiagramContainer {
            return PyDiagramContainer(Arc::new(self.0.generate()));
        });
    }

    pub(crate) fn count(&self, py: Python<'_>) -> usize {
        return py.allow_threads(|| -> usize {
            return self.0.count();
        });
    }

    fn assign_topology(&self, py: Python<'_>, topo: &PyTopology) -> PyResult<PyDiagramContainer> {
        return py.allow_threads(|| -> PyResult<PyDiagramContainer> {
            return Ok(PyDiagramContainer(Arc::new(self.0.assign_topology(&topo.0)?)));
        });
    }

    fn assign_topologies(&self, py: Python<'_>, topos: Vec<PyTopology>) -> PyResult<PyDiagramContainer> {
        return py.allow_threads(|| -> PyResult<PyDiagramContainer> {
            return Ok(PyDiagramContainer(Arc::new(
                self.0
                    .assign_topologies(&topos.iter().map(|t| t.0.clone()).collect_vec())?,
            )));
        });
    }
}
