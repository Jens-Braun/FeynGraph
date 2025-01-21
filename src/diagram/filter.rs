use std::collections::HashMap;
use crate::diagram::Diagram;

pub struct DiagramSelector {
    pub(crate) opi: bool
}

impl Default for DiagramSelector {
    fn default() -> Self { return Self { opi: false } }
}

impl DiagramSelector {
    pub fn select(&self, diag: &Diagram) -> bool {
        return true;
    }

    pub fn set_opi(&mut self) {
        self.opi = !self.opi;
    }
    
    pub(crate) fn get_max_coupling_orders(&self) -> Option<HashMap<String, usize>> { return None; }
}