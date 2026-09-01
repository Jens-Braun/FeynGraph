use std::{path::PathBuf, sync::Arc};

use feyngraph_model::{
    LineStyle, ModelBase, ModelError, ParticleBase, ParticleColor, ParticleDraw, UFOModel, UFOParticle, UFOVertex,
    VertexBase,
};
use feyngraph_qgraf::{ConstValue, QGRAFModel, QGRAFParticle, QGRAFVertex};
use util::{HashMap, IndexMap};

use pyo3::{
    exceptions::PySyntaxError,
    prelude::*,
    types::{PyInt, PyList, PyString},
};

use crate::PyModelError;

#[derive(Debug, Clone, Copy)]
#[pyclass(skip_from_py_object)]
pub enum ModelKind {
    Basic,
    QGRAF,
    UFO,
    Custom,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
#[pyo3(name = "Model")]
pub struct PyModel {
    kind: ModelKind,
    // Common
    particles: IndexMap<String, PyParticle>,
    vertices: IndexMap<String, PyInteraction>,
    couplings: Vec<String>,

    // UFO
    splittings: Option<HashMap<String, HashMap<String, Vec<(usize, usize)>>>>,

    // QGRAF
    supports_sign: bool,

    // Custom
    custom: Option<Arc<Py<PyAny>>>,
}

#[pymethods]
impl PyModel {
    #[new]
    pub(crate) fn __new__() -> Self {
        Self {
            kind: ModelKind::Basic,
            particles: IndexMap::default(),
            vertices: IndexMap::default(),
            couplings: Vec::new(),
            splittings: None,
            supports_sign: false,
            custom: None,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, quiet = false))]
    fn new_qgraf(path: PathBuf, quiet: bool) -> PyResult<Self> {
        let m = QGRAFModel::read(&path)?;
        if !quiet {
            m.emit()?;
        }
        if m.is_some() {
            Ok(m.get().unwrap().into())
        } else {
            Err(PySyntaxError::new_err("Invalid QGRAF model"))
        }
    }

    #[staticmethod]
    fn new_ufo(path: PathBuf) -> PyResult<Self> {
        Ok(UFOModel::parse(&path).map_err(|e| PyModelError::from(e))?.into())
    }

    #[staticmethod]
    fn new_custom<'py>(value: Bound<'py, PyAny>) -> PyResult<Self> {
        Self::from_py(value)
    }

    fn kind(&self) -> ModelKind {
        self.kind
    }

    fn splittings(&self) -> PyResult<HashMap<String, HashMap<String, Vec<(usize, usize)>>>> {
        if let Some(v) = &self.splittings {
            Ok(v.clone())
        } else {
            Err(PySyntaxError::new_err("Splittings are only avaliable in UFO models"))
        }
    }

    fn custom<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        if let Some(custom) = self.custom.as_ref() {
            Ok(custom.clone_ref(py))
        } else {
            Err(PySyntaxError::new_err(
                "Custom objects only exist in custom model types",
            ))
        }
    }

    fn particle(&self, index: usize) -> PyParticle {
        self.particles[index].clone()
    }

    fn particle_by_name(&self, name: String) -> PyResult<PyParticle> {
        return self
            .particles
            .get(&name)
            .cloned()
            .ok_or_else(|| PyModelError::from(ModelError::particle_not_found(&name)).into());
    }

    fn particle_index_by_name(&self, name: String) -> PyResult<usize> {
        return self
            .particles
            .get_index_of(&name)
            .ok_or_else(|| PyModelError::from(ModelError::particle_not_found(&name)).into());
    }

    fn vertex(&self, index: usize) -> PyInteraction {
        self.vertices[index].clone()
    }

    fn particles(&self) -> Vec<PyParticle> {
        self.particles.values().cloned().collect()
    }

    fn vertices(&self) -> Vec<PyInteraction> {
        self.vertices.values().cloned().collect()
    }

    fn couplings(&self) -> Vec<String> {
        self.couplings.clone()
    }

    fn supports_sign(&self) -> bool {
        self.supports_sign
    }
}

impl ModelBase for PyModel {
    type Particle = PyParticle;
    type Vertex = PyInteraction;

    fn particle(&self, index: usize) -> &Self::Particle {
        &self.particles[index]
    }

    fn particle_by_name(&self, name: impl AsRef<str>) -> Result<&Self::Particle, ModelError> {
        return self
            .particles
            .get(name.as_ref())
            .ok_or_else(|| ModelError::particle_not_found(name.as_ref()));
    }

    fn particle_index_by_name(&self, name: impl AsRef<str>) -> Result<usize, ModelError> {
        return self
            .particles
            .get_index_of(name.as_ref())
            .ok_or_else(|| ModelError::particle_not_found(name.as_ref()));
    }

    fn vertex(&self, index: usize) -> &Self::Vertex {
        &self.vertices[index]
    }

    fn particles(&self) -> impl ExactSizeIterator<Item = &Self::Particle> {
        self.particles.values()
    }

    fn vertices(&self) -> impl ExactSizeIterator<Item = &Self::Vertex> {
        self.vertices.values()
    }

    fn couplings(&self) -> impl ExactSizeIterator<Item = impl AsRef<str>> {
        self.couplings.iter()
    }

    fn supports_sign(&self) -> bool {
        self.supports_sign
    }
}

impl From<QGRAFModel> for PyModel {
    fn from(value: QGRAFModel) -> Self {
        let particles = value
            .particles()
            .map(|p| (p.name().to_owned(), p.clone().into()))
            .collect();
        let vertices = value
            .vertices()
            .map(|v| (v.name().to_owned(), v.clone().into()))
            .collect();
        let couplings = value.couplings().map(|s| s.as_ref().to_owned()).collect();
        Self {
            kind: ModelKind::QGRAF,
            particles,
            vertices,
            couplings,
            splittings: None,
            supports_sign: value.supports_sign(),
            custom: None,
        }
    }
}

impl From<UFOModel> for PyModel {
    fn from(value: UFOModel) -> Self {
        let particles = value
            .particles()
            .map(|p| (p.name().to_owned(), p.clone().into()))
            .collect();
        let vertices = value
            .vertices()
            .map(|v| (v.name().to_owned(), v.clone().into()))
            .collect();
        let couplings = value.couplings().map(|s| s.as_ref().to_owned()).collect();
        Self {
            kind: ModelKind::UFO,
            particles,
            vertices,
            couplings,
            splittings: Some(value.splittings().clone()),
            supports_sign: value.supports_sign(),
            custom: None,
        }
    }
}

impl PyModel {
    fn from_py<'py>(value: Bound<'py, PyAny>) -> PyResult<Self> {
        let particles = value
            .call_method0("particles")?
            .try_iter()?
            .map(|v| match v {
                Ok(v) => PyParticle::from_py(v),
                Err(e) => Err(e),
            })
            .map(|v| v.map(|p| (p.name.clone(), p)))
            .collect::<Result<_, _>>()?;
        let vertices = value
            .call_method0("vertices")?
            .try_iter()?
            .map(|v| match v {
                Ok(v) => PyInteraction::from_py(v),
                Err(e) => Err(e),
            })
            .map(|v| v.map(|v| (v.name.clone(), v)))
            .collect::<Result<_, _>>()?;
        let couplings = value.call_method0("couplings")?.extract()?;
        let supports_sign = value.call_method0("supports_sign")?.extract()?;
        Ok(Self {
            kind: ModelKind::Custom,
            particles,
            vertices,
            couplings,
            splittings: None,
            supports_sign,
            custom: Some(Arc::new(value.unbind())),
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
#[pyo3(name = "Particle")]
pub struct PyParticle {
    kind: ModelKind,
    // Common
    name: String,
    anti_name: String,
    id: i64,
    fermi: bool,
    self_anti: bool,
    display_name: Option<String>,
    linestyle: Option<LineStyle>,

    // UFO
    spin: i64,
    color: Option<i64>,

    // QGRAF
    keywords: Option<Vec<String>>,
    constants: Option<HashMap<String, ConstValue>>,

    // Custom
    custom: Option<Arc<Py<PyAny>>>,
}

#[pymethods]
impl PyParticle {
    fn kind(&self) -> ModelKind {
        self.kind
    }

    fn keywords(&self) -> PyResult<Vec<String>> {
        if let Some(k) = &self.keywords {
            Ok(k.clone())
        } else {
            Err(PySyntaxError::new_err(
                "Generic keywords are only available in QGRAF models",
            ))
        }
    }

    fn constants(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        if let Some(v) = &self.constants {
            Ok(v.iter().map(|(k, v)| (k.clone(), const_to_py(py, v))).collect())
        } else {
            Err(PySyntaxError::new_err(
                "Generic constants are only available in QGRAF models",
            ))
        }
    }

    fn custom<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        if let Some(custom) = self.custom.as_ref() {
            Ok(custom.clone_ref(py))
        } else {
            Err(PySyntaxError::new_err(
                "Custom objects only exist in custom model types",
            ))
        }
    }

    fn spin(&self) -> PyResult<i64> {
        match self.kind {
            ModelKind::UFO => Ok(self.spin),
            _ => Err(PySyntaxError::new_err("Spin is only available in UFO models")),
        }
    }

    fn color(&self) -> PyResult<i64> {
        match self.kind {
            ModelKind::UFO => Ok(self.color.unwrap()),
            ModelKind::Custom => self.color.ok_or(PySyntaxError::new_err(
                "Color is only available in custom particles supplying a `color` method",
            )),
            ModelKind::QGRAF | ModelKind::Basic => Err(PySyntaxError::new_err(
                "Color is only supported by UFO models and custom models",
            )),
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn anti_name(&self) -> String {
        self.anti_name.clone()
    }

    fn id(&self) -> i64 {
        self.id
    }

    fn is_self_anti(&self) -> bool {
        self.self_anti
    }

    fn is_fermi(&self) -> bool {
        self.fermi
    }

    fn display_name(&self) -> &str {
        if let Some(s) = &self.display_name {
            s
        } else {
            &self.name
        }
    }

    fn linestyle(&self) -> String {
        if let Some(ls) = self.linestyle {
            ls.to_string()
        } else {
            LineStyle::None.to_string()
        }
    }
}

impl ParticleBase for PyParticle {
    fn name(&self) -> &str {
        &self.name
    }

    fn anti_name(&self) -> &str {
        &self.anti_name
    }

    fn id(&self) -> i64 {
        self.id
    }

    fn is_self_anti(&self) -> bool {
        self.self_anti
    }

    fn is_fermi(&self) -> bool {
        self.fermi
    }
}

impl ParticleDraw for PyParticle {
    fn display_name(&self) -> &str {
        if let Some(s) = &self.display_name {
            s
        } else {
            &self.name
        }
    }

    fn linestyle(&self) -> LineStyle {
        if let Some(ls) = self.linestyle {
            ls
        } else {
            LineStyle::None
        }
    }
}

impl ParticleColor for PyParticle {
    fn color(&self) -> i64 {
        match self.kind {
            ModelKind::UFO => self.color.unwrap(),
            ModelKind::Custom => self
                .color
                .expect("Custom particle objects are required to supply a `color` method to use color"),
            ModelKind::QGRAF | ModelKind::Basic => panic!("Color is only supported by UFO models"),
        }
    }
}

impl From<QGRAFParticle> for PyParticle {
    fn from(value: QGRAFParticle) -> Self {
        Self {
            kind: ModelKind::QGRAF,
            name: value.name,
            anti_name: value.anti_name,
            id: value.id,
            fermi: value.fermi,
            self_anti: value.self_anti,
            display_name: None,
            linestyle: None,
            spin: 0,
            color: None,
            keywords: Some(value.optional_keywords),
            constants: Some(value.constants),
            custom: None,
        }
    }
}

impl From<UFOParticle> for PyParticle {
    fn from(value: UFOParticle) -> Self {
        Self {
            kind: ModelKind::UFO,
            name: value.name,
            anti_name: value.anti_name,
            id: value.pdg_code,
            fermi: value.fermi,
            self_anti: value.self_anti,
            display_name: Some(value.texname),
            linestyle: Some(value.linestyle),
            spin: value.spin,
            color: Some(value.color),
            keywords: None,
            constants: None,
            custom: None,
        }
    }
}

impl PyParticle {
    fn from_py<'py>(value: Bound<'py, PyAny>) -> PyResult<Self> {
        let name = value.call_method0("name")?.extract()?;
        let anti_name = value.call_method0("anti_name")?.extract()?;
        let id = value.call_method0("id")?.extract()?;
        let fermi = value.call_method0("is_fermi")?.extract()?;
        let self_anti = value.call_method0("is_self_anti")?.extract()?;
        let display_name = match value.call_method0("display_name").ok() {
            None => None,
            Some(v) => v.extract()?,
        };
        let linestyle = match value.call_method0("linestyle").ok() {
            None => None,
            Some(v) => Some(LineStyle::from(v.extract::<String>()?)),
        };
        let color = match value.call_method0("color").ok() {
            None => None,
            Some(v) => v.extract()?,
        };
        Ok(Self {
            kind: ModelKind::Custom,
            name,
            anti_name,
            id,
            fermi,
            self_anti,
            display_name,
            linestyle,
            spin: 0,
            color,
            keywords: None,
            constants: None,
            custom: Some(Arc::new(value.unbind())),
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
#[pyo3(name = "Interaction")]
pub struct PyInteraction {
    kind: ModelKind,
    // Common
    name: String,
    particles: Vec<String>,
    fermi_map: Vec<i64>,
    coupling_orders: HashMap<String, usize>,

    // QGRAF
    constants: Option<HashMap<String, ConstValue>>,

    // Custom
    custom: Option<Arc<Py<PyAny>>>,
}

#[pymethods]
impl PyInteraction {
    fn kind(&self) -> ModelKind {
        self.kind
    }

    fn constants(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        if let Some(v) = &self.constants {
            Ok(v.iter().map(|(k, v)| (k.clone(), const_to_py(py, v))).collect())
        } else {
            Err(PySyntaxError::new_err(
                "Generic constants are only available in QGRAF models",
            ))
        }
    }

    fn custom<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        if let Some(custom) = self.custom.as_ref() {
            Ok(custom.clone_ref(py))
        } else {
            Err(PySyntaxError::new_err(
                "Custom objects only exist in custom model types",
            ))
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn particles(&self) -> Vec<String> {
        self.particles.clone()
    }

    fn coupling_orders(&self) -> HashMap<String, usize> {
        self.coupling_orders.clone()
    }

    fn fermi_map(&self, in_ray: usize) -> i64 {
        self.fermi_map[in_ray]
    }
}

impl VertexBase for PyInteraction {
    fn name(&self) -> &str {
        &self.name
    }

    fn particles(&self) -> &[impl AsRef<str>] {
        &self.particles
    }

    fn coupling_orders(&self) -> &HashMap<String, usize> {
        &self.coupling_orders
    }

    fn fermi_map(&self, in_ray: usize) -> i64 {
        self.fermi_map[in_ray]
    }
}

impl From<QGRAFVertex> for PyInteraction {
    fn from(value: QGRAFVertex) -> Self {
        Self {
            kind: ModelKind::QGRAF,
            name: value.name,
            particles: value.particles,
            fermi_map: value.fermi_map,
            coupling_orders: value.couplings,
            constants: Some(value.constants),
            custom: None,
        }
    }
}

impl From<UFOVertex> for PyInteraction {
    fn from(value: UFOVertex) -> Self {
        Self {
            kind: ModelKind::UFO,
            name: value.name,
            particles: value.particles,
            fermi_map: value.fermi_map,
            coupling_orders: value.coupling_orders,
            constants: None,
            custom: None,
        }
    }
}

impl PyInteraction {
    fn from_py<'py>(value: Bound<'py, PyAny>) -> PyResult<Self> {
        let name = value.call_method0("name")?.extract()?;
        let particles: Vec<String> = value.call_method0("particles")?.extract()?;
        let coupling_orders = match value.call_method0("coupling_orders") {
            Ok(v) => v.extract()?,
            Err(_) => HashMap::default(),
        };
        let fermi_map = particles
            .iter()
            .enumerate()
            .map(|(i, _)| -> PyResult<i64> { Ok(value.call_method1("fermi_map", (i,))?.extract()?) })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            kind: ModelKind::Custom,
            name,
            particles,
            fermi_map,
            coupling_orders,
            constants: None,
            custom: Some(Arc::new(value.unbind())),
        })
    }
}

fn const_to_py(py: Python<'_>, v: &ConstValue) -> Py<PyAny> {
    match v {
        ConstValue::Int(i) => PyInt::new(py, i).into_any(),
        ConstValue::String(s) => PyString::new(py, s).into_any(),
        ConstValue::List(l) => PyList::new(py, l.iter().map(|v| const_to_py(py, v)))
            .unwrap()
            .into_any(),
    }
    .unbind()
}
