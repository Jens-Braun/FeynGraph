#![allow(dead_code)]
#![allow(clippy::needless_return, clippy::result_large_err, clippy::needless_range_loop)]

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

pub mod topology;
pub mod model;
pub(crate) mod util;

#[cfg(feature = "python-bindings")]
#[pymodule]
#[allow(non_snake_case)]
fn feyngraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    let topology_submodule = PyModule::new(m.py(), "topology")?;
    topology_submodule.add_class::<model::TopologyModel>()?;
    topology_submodule.add_class::<topology::Topology>()?;
    topology_submodule.add_class::<topology::TopologyContainer>()?;
    topology_submodule.add_class::<topology::TopologyGenerator>()?;
    topology_submodule.add_class::<topology::filter::TopologySelector>()?;
    m.add_submodule(&topology_submodule)?;
    m.add_class::<model::Model>()?;
    return Ok(());
}