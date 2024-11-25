use std::ops::Range;
use itertools::Itertools;
use pyo3::prelude::*;
use crate::topology::filter::SelectionCriterion::*;
use crate::topology::Topology;

#[cfg_attr(feature = "python-bindings", pyclass)]
#[derive(Debug, Clone)]
pub enum SelectionCriterion {
    NodeDegree {degree: usize, selection: Vec<usize>},
    NodePartition(Vec<(usize, usize)>),
    OPIComponents(usize),
}

#[cfg_attr(feature = "python-bindings", pyclass)]
#[derive(Debug, Clone)]
pub struct TopologySelector {
    criteria: Vec<SelectionCriterion>
}

#[cfg_attr(feature = "python-bindings", pymethods)]
impl TopologySelector {
    
    #[new]
    fn new() -> Self {
        return Self::default();
    }
    
    pub fn add_criterion(&mut self, criterion: SelectionCriterion) {
        self.criteria.push(criterion);
    }

    pub fn add_node_degree(&mut self, degree: usize, selection: usize) {
        if let Some(NodeDegree { degree: _, selection: inner_selection }) =
            &mut self.criteria.iter_mut().find(|criterion| {
                return if let NodeDegree { degree: inner_degree, .. } = criterion {
                    if *inner_degree == degree { true } else { false }
                } else { false }
            }) {
            inner_selection.push(selection)
        } else {
            self.criteria.push(NodeDegree { degree, selection: vec![selection] });
        }
    }

    pub fn add_node_degree_selection(&mut self, degree: usize, mut selection: Vec<usize>) {
        if let Some(NodeDegree { degree: _, selection: inner_selection }) =
            &mut self.criteria.iter_mut().find(|criterion| {
                return if let NodeDegree { degree: inner_degree, .. } = criterion {
                    if *inner_degree == degree { true } else { false }
                } else { false }
            }) {
            inner_selection.append(&mut selection)
        } else {
            self.criteria.push(NodeDegree { degree, selection });
        }
    }

    pub fn add_node_partition(&mut self, partition: Vec<(usize, usize)>) {
        self.criteria.push(NodePartition(partition));
    }

    pub fn add_opi_count(&mut self, count: usize) {
        self.criteria.push(OPIComponents(count));
    }
}

impl TopologySelector {
    pub fn add_node_degree_range(&mut self, degree: usize, selection: Range<usize>) {
        if let Some(NodeDegree { degree: _, selection: inner_selection }) =
            &mut self.criteria.iter_mut().find(|criterion| {
                return if let NodeDegree { degree: inner_degree, .. } = criterion {
                    if *inner_degree == degree { true } else { false }
                } else { false }
            }) {
            inner_selection.append(&mut selection.collect_vec())
        } else {
            self.criteria.push(NodeDegree { degree, selection: selection.collect_vec() });
        }
    }
    
    pub(crate) fn select(&self, topo: &Topology) -> bool {
        for criterion in &self.criteria {
            if false == match criterion {
                NodeDegree { degree, selection} => {
                    selection.iter().map(|count|  {
                        topo.node_degrees.iter().filter(|deg| **deg == *degree).count() == *count
                    }).any(|x| x)
                },
                NodePartition(partition) => {
                    partition.iter().map(|(degree, count)|  {
                        topo.node_degrees.iter().filter(|deg| **deg == *degree).count() == *count
                    }).all(|x| x)
                }
                OPIComponents(count) => {
                    topo.count_opi() == *count
                }
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
        return Self { criteria: Vec::new()};
    }
}