use model::VertexBase;
use util::HashMap;

use crate::ConstValue;

#[derive(Debug, Clone, PartialEq)]
pub struct QGRAFVertex {
    pub name: String,
    pub particles: Vec<String>,
    pub couplings: HashMap<String, usize>,
    pub constants: HashMap<String, ConstValue>,
    pub fermi_map: Vec<i64>,
}

impl QGRAFVertex {
    pub(crate) fn new(
        name: impl Into<String>,
        particles: Vec<String>,
        couplings: HashMap<String, usize>,
        constants: HashMap<String, ConstValue>,
        fermi_map: Vec<i64>,
    ) -> Self {
        Self {
            name: name.into(),
            particles,
            couplings,
            constants,
            fermi_map,
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

    fn fermi_map(&self, in_ray: usize) -> i64 {
        self.fermi_map[in_ray]
    }
}
