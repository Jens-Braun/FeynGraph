#![allow(dead_code)]

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

pub mod topology;
pub mod model;
pub(crate) mod util;

#[cfg(feature = "python-bindings")]
#[pymodule]
#[allow(non_snake_case)]
fn FeynGraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let topology_submodule = PyModule::new(m.py(), "topology")?;
    topology_submodule.add_class::<model::TopologyModel>()?;
    topology_submodule.add_class::<topology::Topology>()?;
    topology_submodule.add_class::<topology::TopologyContainer>()?;
    topology_submodule.add_class::<topology::TopologyGenerator>()?;
    topology_submodule.add_class::<topology::filter::TopologySelector>()?;
    m.add_submodule(&topology_submodule)?;
    return Ok(());
}