use model::VertexBase;
use util::HashMap;

use crate::ConstValue;

#[derive(Debug, Clone, PartialEq)]
pub struct QGRAFVertex {
    name: String,
    particles: Vec<String>,
    couplings: HashMap<String, usize>,
    constants: HashMap<String, ConstValue>,
    spin_map: Vec<isize>,
}

impl QGRAFVertex {
    pub(crate) fn new(
        name: impl Into<String>,
        particles: Vec<String>,
        couplings: HashMap<String, usize>,
        constants: HashMap<String, ConstValue>,
        spin_map: Vec<isize>,
    ) -> Self {
        Self {
            name: name.into(),
            particles,
            couplings,
            constants,
            spin_map,
        }
    }

    pub fn consts(&self) -> &HashMap<String, ConstValue> {
        &self.constants
    }
}

impl VertexBase for QGRAFVertex {
    fn name(&self) -> &str {
        &self.name
    }

    fn particles(&self) -> &[impl AsRef<str>] {
        &self.particles
    }

    fn coupling_orders(&self) -> &HashMap<String, usize> {
        &self.couplings
    }

    fn fermi_map(&self, in_ray: usize) -> usize {
        self.spin_map[in_ray].try_into().unwrap()
    }
}
