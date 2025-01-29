use std::fmt::Formatter;
use std::ops::Range;
use std::sync::Arc;
use itertools::Itertools;
use crate::topology::Topology;


/// A struct that decides whether a topology is to be kept or discarded. Only topologies for which
/// `selector.select(&topology) == true` are kept. Multiple criteria can be added, the selector will
/// then select diagrams which satisfy any of them.
#[derive(Clone)]
pub struct TopologySelector {
    /// Only keep topologies for which the count of nodes with the given `degree` is in `counts`, specified as a list of
    /// `(degree, counts)`
    pub(crate) node_degrees: Vec<(usize, Vec<usize>)>,
    /// Only keep topologies with the specified node partition, specified as a list of `(degree, count)`
    pub(crate) node_partition: Vec<Vec<(usize, usize)>>,
    /// Only keep topologies with the specified number of one-particle-irreducible components
    pub(crate) opi_components: Vec<usize>,
    /// Only keep topologies for which the given custom function returns `true`
    pub(crate) custom_functions: Vec<Arc<dyn Fn(&Topology) -> bool + Sync + Send>>
}

impl TopologySelector {
    
    /// Create a new [TopologySelector] which selects every diagram.
    pub fn new() -> Self {
        return Self {
            node_degrees: Vec::new(),
            node_partition: Vec::new(),
            opi_components: Vec::new(),
            custom_functions: Vec::new()
        };
    }
    
    /// Add a criterion to keep only diagrams with `selection` number of nodes with `degree`.
    pub fn add_node_degree(&mut self, degree: usize, selection: usize) {
        if let Some((_, counts)) =
            &mut self.node_degrees.iter_mut().find(|(constrained_degree, _)| *constrained_degree == degree) {
            counts.push(selection)
        } else {
            self.node_degrees.push((degree, vec![selection]));
        }
    }

    /// Add a criterion to keep only diagrams for which the number of nodes with `degree` is contained
    /// in `selection`.
    pub fn add_node_degree_list(&mut self, degree: usize, mut selection: Vec<usize>) {
        if let Some((_, counts)) =
            &mut self.node_degrees.iter_mut().find(|(constrained_degree, _)| *constrained_degree == degree) {
            counts.append(&mut selection)
        } else {
            self.node_degrees.push((degree, selection));
        }
    }

    /// Add a criterion to keep only diagrams for which the number of nodes with `degree` is contained
    /// in the range `selection`.
    pub fn add_node_degree_range(&mut self, degree: usize, selection: Range<usize>) {
        if let Some((_, counts)) =
            &mut self.node_degrees.iter_mut().find(|(constrained_degree, _)| *constrained_degree == degree) {
            counts.append(&mut selection.collect_vec());
        } else {
            self.node_degrees.push((degree, selection.collect_vec()));
        }
    }

    /// Add a criterion to keep only diagrams with the node partition given by `partition`. The node partition is the
    /// set of counts of nodes with given degrees, e.g. the partition
    /// ```rust
    ///     use feyngraph::topology::filter::TopologySelector;
    /// let mut selector = TopologySelector::new();
    /// selector.add_node_partition(vec![(3, 4), (4, 1)]);
    /// ```
    /// selects only topologies which include _exactly_ three nodes of degree 3 and one node of degree 4.
    pub fn add_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.node_partition.push(partition);
    }

    /// Add a criterion to keep only diagrams with `count` one-particle-irreducible components.
    pub fn add_opi_count(&mut self, count: usize) {
        self.opi_components.push(count);
    }

    /// Add a criterion to keep only diagrams for which the given function returns `true`.
    pub fn add_custom_function(&mut self, function: Arc<dyn Fn(&Topology) -> bool + Sync + Send>) {
        self.custom_functions.push(function);
    }

    /// Clear the previously added criteria.
    pub fn clear_criteria(&mut self) {
        self.node_degrees.clear();
        self.node_partition.clear();
        self.opi_components.clear();
        self.custom_functions.clear();
    }
    
    pub(crate) fn select(&self, topo: &Topology) -> bool {
        return self.select_node_degrees(topo)
            && self.select_node_partition(topo)
            && self.select_opi_components(topo)
            && self.select_custom_criteria(topo);
    }

    fn select_node_degrees(&self, topo: &Topology) -> bool {
        return self.node_degrees.iter().all(|(degree, counts)| {
            let topo_count = topo.nodes.iter().filter(|node| node.degree == *degree).count();
            if counts.iter().map(|count|  topo_count == *count).any(|x| x) {
                return true;
            }
            return false;
        });
    }

    fn select_node_partition(&self, topo: &Topology) -> bool {
        for partition in &self.node_partition {
            if partition.iter().map(|(degree, count)|  {
                topo.nodes.iter().filter(|node| node.degree == *degree).count() != *count
            }).any(|x| x) {
                return false;
            }
        }
        return true;
    }

    fn select_opi_components(&self, topo: &Topology) -> bool {
        if self.opi_components.is_empty() { return true; }
        return self.opi_components.iter().any(|opi_count| *opi_count == topo.count_opi_componenets());
    }

    fn select_custom_criteria(&self, topo: &Topology) -> bool {
        if self.custom_functions.is_empty() { return true; }
        for custom_function in &self.custom_functions {
            if custom_function(topo) { return true; }
        }
        return false;
    }

    pub(crate) fn select_partition(&self, partition: Vec<(usize, usize)>) -> bool {
        for selected_partition in &self.node_partition {
            if !selected_partition.iter().map(|(selected_degree, selected_count)| {
                partition.iter().find_map(|(degree, count)|
                    if *degree == *selected_degree {Some(*count)} else {None}
                ) == Some(*selected_count)
            }).all(|x| x) {
                return false;
            }
        }
        for (selected_degree, counts) in &self.node_degrees {
            if !counts.contains(&partition.iter().find_map(|(degree, count)|
                if *degree == *selected_degree {Some(*count)} else {None}
            ).unwrap_or(0)) {
                return false;
            }
        }
        return true;
    }
}

impl std::fmt::Debug for TopologySelector {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        return fmt.debug_struct("TopologySelector")
            .field("node_degrees", &self.node_degrees)
            .field("node_partition", &self.node_partition)
            .field("opi_components", &self.opi_components)
            .field("custom_functions", &format!("{} custom functions", self.custom_functions.len()))
            .finish();
    }
}

impl Default for TopologySelector {
    fn default() -> Self {
        return TopologySelector::new();
    }
}