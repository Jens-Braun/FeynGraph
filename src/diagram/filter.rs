use crate::diagram::Diagram;
use crate::diagram::view::DiagramView;
use crate::model::Model;
use crate::topology::Topology;
use crate::topology::filter::TopologySelector;
use crate::util::HashMap;
use itertools::Itertools;
use std::sync::Arc;

/// A struct that decides whether a diagram is to be kept or discarded. Only diagrams for which
/// `selector.select(&topology) == true` are kept. Multiple criteria can be added, the selector will
/// then select diagrams which satisfy any of them.
///
/// # Examples
/// ```rust
/// use feyngraph::DiagramSelector;
/// let mut s = DiagramSelector::new();
/// s.select_on_shell();
/// s.select_self_loops(0);
/// s.select_coupling_power("NP", 1);
/// ```
#[derive(Clone)]
pub struct DiagramSelector {
    /// Only keep diagrams with the specified number of one-particle-irreducible components
    pub(crate) opi_components: Vec<usize>,
    /// Only keep diagrams with the specified number of self-loops
    pub(crate) self_loops: Vec<usize>,
    /// Only keep diagrams with no self-energy insertions on external legs.
    pub(crate) on_shell: bool,
    /// Only keep diagrams for which the power of the given coupling is contained in the list of specified powers
    pub(crate) coupling_powers: HashMap<String, Vec<usize>>,
    /// Only keep diagrams for which the number of propagators of the given field is in the list of specified counts
    pub(crate) propagator_counts: HashMap<String, Vec<usize>>,
    /// Only keep diagrams for which the number of vertices of the given fields is in the list of specified counts
    pub(crate) vertex_counts: HashMap<Vec<String>, Vec<usize>>,
    /// Only keep diagrams for which the number of vertices with the given degrees is in the list of specified counts
    pub(crate) vertex_degree_counts: Vec<(usize, Vec<usize>)>,
    /// Only keep diagrams for which the given custom function returns `true`
    #[allow(clippy::type_complexity)]
    pub(crate) custom_functions: Vec<Arc<dyn Fn(Arc<Model>, Arc<Vec<String>>, &Diagram) -> bool + Sync + Send>>,
    /// Same as [custom_functions], but used when the [DiagramSelector] is cast to a [TopologySelector]
    #[allow(clippy::type_complexity)]
    pub(crate) topology_functions: Vec<Arc<dyn Fn(&Topology) -> bool + Sync + Send>>,
}

impl Default for DiagramSelector {
    fn default() -> Self {
        return Self::new();
    }
}

impl DiagramSelector {
    pub fn new() -> Self {
        return Self {
            opi_components: Vec::new(),
            self_loops: Vec::new(),
            on_shell: false,
            coupling_powers: HashMap::default(),
            propagator_counts: HashMap::default(),
            vertex_counts: HashMap::default(),
            vertex_degree_counts: Vec::new(),
            custom_functions: Vec::new(),
            topology_functions: Vec::new(),
        };
    }

    /// Add a criterion to keep only diagrams with `count` one-particle-irreducible components.
    pub fn select_opi_components(&mut self, count: usize) {
        self.opi_components.push(count);
    }

    /// Add a criterion to only keep diagrams with `count` self loops. A self-loop is defined as a propagator which ends
    /// on the same vertex it started on.
    pub fn select_self_loops(&mut self, count: usize) {
        self.self_loops.push(count);
    }

    /// Toggle the on-shell criterion. If true, only diagrams with no self-energy insertions on external legs are
    /// kept. This implementation considers internal propagators carrying a single external momentum and no loop
    /// momentum, which is equivalent to a self-energy insertion on an external edge.
    pub fn select_on_shell(&mut self) {
        self.on_shell = !self.on_shell;
    }

    /// Add a criterion to only keep diagrams which have power `power` in the given coupling `coupling`.
    pub fn select_coupling_power(&mut self, coupling: &str, power: usize) {
        if let Some(powers) = self.coupling_powers.get_mut(coupling) {
            if !powers.contains(&power) {
                powers.push(power);
            }
        } else {
            self.coupling_powers.insert(coupling.to_string(), vec![power]);
        }
    }

    /// Add a criterion to only keep diagrams which contain `count` propagators of the field `particle`.
    pub fn select_propagator_count(&mut self, particle: &str, count: usize) {
        if let Some(powers) = self.propagator_counts.get_mut(particle) {
            if !powers.contains(&count) {
                powers.push(count);
            }
        } else {
            self.propagator_counts.insert(particle.to_string(), vec![count]);
        }
    }

    /// Add a criterion to only keep diagrams which contain `count` vertices of the fields `particles`.
    pub fn select_vertex_count(&mut self, particles: Vec<String>, count: usize) {
        let particles_sorted = particles.into_iter().sorted_unstable().collect_vec();
        if let Some(powers) = self.vertex_counts.get_mut(&particles_sorted) {
            if !powers.contains(&count) {
                powers.push(count);
            }
        } else {
            self.vertex_counts.insert(particles_sorted, vec![count]);
        }
    }

    /// Add a criterion to only keep diagrams for which the power of the coupling `coupling` is contained in `powers`.
    pub fn select_coupling_power_list(&mut self, coupling: &str, mut powers: Vec<usize>) {
        if let Some(existing_powers) = self.coupling_powers.get_mut(coupling) {
            existing_powers.append(&mut powers);
        } else {
            self.coupling_powers.insert(coupling.to_string(), powers);
        }
    }

    /// Add a criterion to only keep diagrams which contains `count` vertices of degree `degree`.
    pub fn select_vertex_degree(&mut self, degree: usize, count: usize) {
        if let Some((_, counts)) = self.vertex_degree_counts.iter_mut().find(|(d, _)| *d == degree) {
            if !counts.contains(&count) {
                counts.push(count);
            }
        } else {
            self.vertex_degree_counts.push((degree, vec![count]));
        }
    }

    /// Custom function handed to the [`TopologyGenerator`](crate::topology::TopologyGenerator) used to generate topologies for a
    /// [`DiagramGenerator`](super::DiagramGenerator).
    pub fn add_topology_function(&mut self, function: Arc<dyn Fn(&Topology) -> bool + Sync + Send>) {
        self.topology_functions.push(function);
    }

    /// Add a criterion to keep only diagrams for which the given function returns `true`.
    pub fn add_custom_function(&mut self, function: Arc<dyn Fn(&DiagramView) -> bool + Sync + Send>) {
        self.custom_functions.push(Arc::new(
            move |model: Arc<Model>, momentum_labels: Arc<Vec<String>>, diag: &Diagram| -> bool {
                function(&DiagramView {
                    model: model.as_ref(),
                    momentum_labels: momentum_labels.as_ref(),
                    diagram: diag,
                })
            },
        ));
    }

    /// Add a custom function which takes the internal representation of a diagram as input.
    #[allow(clippy::type_complexity)]
    pub(crate) fn add_unwrapped_custom_function(
        &mut self,
        function: Arc<dyn Fn(Arc<Model>, Arc<Vec<String>>, &Diagram) -> bool + Sync + Send>,
    ) {
        self.custom_functions.push(function);
    }

    pub(crate) fn select(&self, model: Arc<Model>, momentum_labels: Arc<Vec<String>>, diag: &Diagram) -> bool {
        let view = DiagramView {
            model: model.as_ref(),
            momentum_labels: momentum_labels.as_ref(),
            diagram: diag,
        };
        return self.query_opi_components(view.diagram)
            && self.query_custom_criteria(model.clone(), momentum_labels.clone(), diag)
            && self.query_coupling_powers(&view)
            && self.query_propagator_counts(&view)
            && self.query_vertex_counts(&view);
    }

    fn query_opi_components(&self, diag: &Diagram) -> bool {
        if self.opi_components.is_empty() {
            return true;
        }
        return self
            .opi_components
            .iter()
            .any(|opi_count| *opi_count == diag.count_opi_components());
    }

    fn query_custom_criteria(&self, model: Arc<Model>, momentum_labels: Arc<Vec<String>>, diag: &Diagram) -> bool {
        if self.custom_functions.is_empty() {
            return true;
        }
        for custom_function in &self.custom_functions {
            if custom_function(model.clone(), momentum_labels.clone(), diag) {
                return true;
            }
        }
        return false;
    }

    fn query_coupling_powers(&self, view: &DiagramView) -> bool {
        if self.coupling_powers.is_empty() {
            return true;
        }
        return self.coupling_powers.iter().all(|(coupling, powers)| {
            powers.iter().any(|power| {
                view.vertices()
                    .map(|v| *v.interaction().coupling_orders.get(coupling).unwrap_or(&0))
                    .sum::<usize>()
                    == *power
            })
        });
    }

    fn query_propagator_counts(&self, view: &DiagramView) -> bool {
        if self.propagator_counts.is_empty() {
            return true;
        }
        return self.propagator_counts.iter().all(|(particle, counts)| {
            counts.iter().any(|count| {
                view.propagators()
                    .filter(|prop| *prop.particle().name() == *particle)
                    .count()
                    == *count
            })
        });
    }

    fn query_vertex_counts(&self, view: &DiagramView) -> bool {
        if self.vertex_counts.is_empty() {
            return true;
        }
        return self.vertex_counts.iter().all(|(particles, counts)| {
            counts
                .iter()
                .any(|count| view.vertices().filter(|v| v.match_particles(particles)).count() == *count)
        });
    }

    pub(crate) fn get_max_coupling_orders(&self) -> Option<HashMap<String, usize>> {
        return if self.coupling_powers.is_empty() {
            None
        } else {
            Some(
                self.coupling_powers
                    .iter()
                    .map(|(coupling, powers)| (coupling.clone(), powers.iter().max().cloned().unwrap()))
                    .collect(),
            )
        };
    }

    pub(crate) fn as_topology_selector(&self) -> TopologySelector {
        return TopologySelector {
            node_degrees: self.vertex_degree_counts.clone(),
            node_partition: Vec::new(),
            opi_components: self.opi_components.clone(),
            self_loops: self.self_loops.clone(),
            on_shell: self.on_shell,
            custom_functions: self.topology_functions.clone(),
        };
    }
}
