use crate::topology::Topology;

pub struct TopologySelector {
    
}

impl TopologySelector {
    pub fn select(&self, topo: &Topology) -> bool {
        todo!();
    }
}

impl Default for TopologySelector {
    fn default() -> Self {
        return Self {};
    }
}