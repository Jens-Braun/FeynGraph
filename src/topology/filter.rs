use std::fmt::Formatter;
use std::ops::Range;
use std::sync::Arc;
use itertools::Itertools;
use crate::topology::filter::SelectionCriterion::*;
use crate::topology::Topology;
use crate::diagram::filter::DiagramSelector;

/// Possible criteria, which are used by a [TopologySelector] to decide whether a diagram is kept.
#[derive(Clone)]
pub enum SelectionCriterion {
    /// Only keep topologies for which the count of nodes with `degree` is in `selection`
    NodeDegree {degree: usize, selection: Vec<usize>},
    /// Only keep topologies with the specified node partition, specified as a list of (degree, count)
    NodePartition(Vec<(usize, usize)>),
    /// Only keep topologies with the specified number of one-particle-irreducible components
    OPIComponents(usize),
    /// Only keep topologies for which the custom function returns `true`
    CustomCriterion(Arc<dyn Fn(&Topology) -> bool + Sync + Send>)
}

impl std::fmt::Debug for SelectionCriterion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionCriterion::NodeDegree {degree, selection} => { 
                f.debug_tuple("NodeDegree").field(degree).field(selection).finish()
            },
            SelectionCriterion::NodePartition(partition) => { 
                f.debug_tuple("NodePartition").field(partition).finish() 
            },
            SelectionCriterion::OPIComponents(n) => { 
                f.debug_tuple("OPIComponents").field(n).finish() 
            },
            SelectionCriterion::CustomCriterion(_) => { f.debug_tuple("CustomCriterion").finish() }
        }
    }
}

/// A struct that decides whether a topology is to be kept or discarded. Only topologies for which
/// `selector.select(&topology) == true` are kept. Multiple criteria can be added, the selector will
/// then select diagrams which satisfy any of them.
#[derive(Debug, Clone)]
pub struct TopologySelector {
    criteria: Vec<SelectionCriterion>
}

impl TopologySelector {
    
    /// Create a new [TopologySelector] which selects every diagram.
    pub fn new() -> Self {
        return Self {criteria: Vec::new()};
    }
    
    /// Add a [SelectionCriterion] to the selector.
    pub fn add_criterion(&mut self, criterion: SelectionCriterion) {
        self.criteria.push(criterion);
    }

    /// Add a criterion to keep only diagrams with `selection` number of nodes with `degree`.
    pub fn add_node_degree(&mut self, degree: usize, selection: usize) {
        if let Some(NodeDegree { degree: _, selection: inner_selection }) =
            &mut self.criteria.iter_mut().find(|criterion| {
                return if let NodeDegree { degree: inner_degree, .. } = criterion {
                    *inner_degree == degree
                } else { false }
            }) {
            inner_selection.push(selection)
        } else {
            self.criteria.push(NodeDegree { degree, selection: vec![selection] });
        }
    }

    /// Add a criterion to keep only diagrams for which the number of nodes with `degree` is contained
    /// in `selection`.
    pub fn add_node_degree_selection(&mut self, degree: usize, mut selection: Vec<usize>) {
        if let Some(NodeDegree { degree: _, selection: inner_selection }) =
            &mut self.criteria.iter_mut().find(|criterion| {
                return if let NodeDegree { degree: inner_degree, .. } = criterion {
                    *inner_degree == degree
                } else { false }
            }) {
            inner_selection.append(&mut selection)
        } else {
            self.criteria.push(NodeDegree { degree, selection });
        }
    }

    /// Add a criterion to keep only diagrams with the node partition given by `partition`.
    pub fn add_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.criteria.push(NodePartition(partition));
    }

    /// Add a criterion to keep only diagrams with `count` one-particle-irreducible components.
    pub fn add_opi_count(&mut self, count: usize) {
        self.criteria.push(OPIComponents(count));
    }

    /// Clear the previously selected critera
    pub fn clear_criteria(&mut self) {
        self.criteria.clear();
    }
    
    /// Add a criterion to keep only diagrams for which the number of nodes with `degree` is contained
    /// in the range `selection`.
    pub fn add_node_degree_range(&mut self, degree: usize, selection: Range<usize>) {
        if let Some(NodeDegree { degree: _, selection: inner_selection }) =
            &mut self.criteria.iter_mut().find(|criterion| {
                return if let NodeDegree { degree: inner_degree, .. } = criterion {
                    *inner_degree == degree
                } else { false }
            }) {
            inner_selection.append(&mut selection.collect_vec())
        } else {
            self.criteria.push(NodeDegree { degree, selection: selection.collect_vec() });
        }
    }
    
    pub(crate) fn select(&self, topo: &Topology) -> bool {
        for criterion in &self.criteria {
            if !match criterion {
                NodeDegree { degree, selection} => {
                    selection.iter().map(|count|  {
                        topo.nodes.iter().filter(|node| node.degree == *degree).count() == *count
                    }).any(|x| x)
                },
                NodePartition(partition) => {
                    partition.iter().map(|(degree, count)|  {
                        topo.nodes.iter().filter(|node| node.degree == *degree).count() == *count
                    }).all(|x| x)
                }
                OPIComponents(count) => {
                    topo.count_opi() == *count
                }
                CustomCriterion(f) => f(topo),
            } {
                return false;
            }
        }
        return true;
    }
    
    pub(crate) fn select_partition(&self, partition: Vec<(usize, usize)>) -> bool {
        for criterion in &self.criteria {
            match criterion {
                NodePartition(selected_partition) => {
                    if !selected_partition.iter().map(|(selected_degree, selected_count)| {
                        partition.iter().find_map(|(degree, count)|
                            if *degree == *selected_degree {Some(*count)} else {None}
                        ) == Some(*selected_count)
                    }).all(|x| x) {
                        return false;
                    }
                }
                NodeDegree { degree: selected_degree, selection } => {
                    if !selection.contains(&partition.iter().find_map(|(degree, count)|
                        if *degree == *selected_degree {Some(*count)} else {None}
                    ).unwrap_or(0)) {
                        return false;
                    }
                }
                _ => (),
            }
        }
        return true;
    }
    
}

impl Default for TopologySelector {
    fn default() -> Self {
        return Self { criteria: vec![OPIComponents(1)]};
    }
}

impl From<&DiagramSelector> for TopologySelector {
    fn from(diagram_selector: &DiagramSelector) -> Self {
        let mut selector =  TopologySelector::new();
        if diagram_selector.opi {
            selector.add_opi_count(1);
        }
        return selector;
    }
}