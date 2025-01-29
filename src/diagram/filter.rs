use std::collections::HashMap;
use std::sync::Arc;
use crate::diagram::Diagram;
use crate::topology::filter::TopologySelector;
use crate::topology::Topology;

/// A struct that decides whether a diagram is to be kept or discarded. Only diagrams for which
/// `selector.select(&topology) == true` are kept. Multiple criteria can be added, the selector will
/// then select diagrams which satisfy any of them.
#[derive(Clone)]
pub struct DiagramSelector {
    /// Only keep diagrams with the specified number of one-particle-irreducible components
    pub(crate) opi_components: Vec<usize>,
    /// Only keep diagrams for which the power of the given coupling is contained in the list of specified powers
    pub(crate) coupling_powers: HashMap<String, Vec<usize>>,
    /// Only keep topologies for which the given custom function returns `true`
    custom_functions: Vec<Arc<dyn Fn(&Diagram) -> bool + Sync + Send>>,
    /// Same as [custom_functions], but used when the [DiagramSelector] is cast to a [TopologySelector]
    pub(crate) topology_functions: Vec<Arc<dyn Fn(&Topology) -> bool + Sync + Send>>
}

impl Default for DiagramSelector {
    fn default() -> Self { return Self {
        opi_components: Vec::new(),
        coupling_powers: HashMap::new(),
        custom_functions: Vec::new(),
        topology_functions: Vec::new()
    } }
}

impl DiagramSelector {

    /// Add a criterion to keep only diagrams with `count` one-particle-irreducible components.
    pub fn add_opi_count(&mut self, count: usize) {
        self.opi_components.push(count);
    }

    /// Add a criterion to only keep diagrams which have power `power` in the given coupling `coupling`
    pub fn add_coupling_power(&mut self, coupling: &str, power: usize) {
        if let Some(powers) = self.coupling_powers.get_mut(coupling) {
            if !powers.contains(&power) {
                powers.push(power);
            }
        } else {
            self.coupling_powers.insert(coupling.to_string(), vec![power]);
        }
    }

    /// Add a criterion to only keep diagrams for which the power of the coupling `coupling` is contained in `powers`
    pub fn add_coupling_power_list(&mut self, coupling: &str, mut powers: Vec<usize>) {
        if let Some(existing_powers) = self.coupling_powers.get_mut(coupling) {
            existing_powers.append(&mut powers);
        } else {
            self.coupling_powers.insert(coupling.to_string(), powers);
        }
    }

    /// Custom function handed to the [TopologyGenerator] used to generate topologies for a [DiagramGenerator]
    pub fn add_topology_function(&mut self, function: Arc<dyn Fn(&Topology) -> bool + Sync + Send>) {
        self.topology_functions.push(function);
    }

    /// Add a criterion to keep only diagrams for which the given function returns `true`.
    pub fn add_custom_function(&mut self, function: Arc<dyn Fn(&Diagram) -> bool + Sync + Send>) {
        self.custom_functions.push(function);
    }

    pub(crate) fn select(&self, diag: &Diagram) -> bool {
        return self.select_opi_components(diag) && self.select_custom_criteria(diag);
    }

    fn select_opi_components(&self, diag: &Diagram) -> bool {
        if self.opi_components.is_empty() { return true; }
        return self.opi_components.iter().any(|opi_count| *opi_count == diag.count_opi_components());
    }

    fn select_custom_criteria(&self, diag: &Diagram) -> bool {
        if self.custom_functions.is_empty() { return true; }
        for custom_function in &self.custom_functions {
            if custom_function(diag) { return true; }
        }
        return false;
    }
    
    pub(crate) fn get_max_coupling_orders(&self) -> Option<HashMap<String, usize>> {
        return if self.coupling_powers.is_empty() {
            None
        } else {
            Some(
                self.coupling_powers.iter().map(
                    |(coupling, powers)| (coupling.clone(), powers.iter().max().cloned().unwrap())
                ).collect()
            )
        }
    }

    pub(crate) fn as_topology_selector(&self) -> TopologySelector {
        return TopologySelector {
            node_degrees: Vec::new(),
            node_partition: Vec::new(),
            opi_components: self.opi_components.clone(),
            custom_functions: self.topology_functions.clone(),
        };
    }
}