use crate::diagram::components::{AssignPropagator, AssignVertex, VertexClassification};
use crate::diagram::filter::DiagramSelector;
use crate::diagram::{Diagram, DiagramContainer};
use crate::model::Model;
use crate::topology::Topology;
use crate::util::HashMap;
use crate::util::generate_permutations;
use itertools::{FoldWhile, Itertools};
use std::cmp::Ordering;
use std::sync::Arc;

pub(crate) struct AssignWorkspace<'a> {
    pub(crate) topology: &'a Topology,
    // The workspace carries an Arc of the model, such that the model can be handed to a Python function (which is not
    // allowed to take an object with a Rust lifetime) in the custom filter functions
    pub(crate) model: Arc<Model>,
    pub(crate) momentum_labels: Arc<Vec<String>>,
    pub(crate) selector: &'a DiagramSelector,
    pub(crate) incoming_particles: &'a Vec<usize>,
    pub(crate) outgoing_particles: &'a Vec<usize>,
    pub(crate) diagram_buffer: DiagramContainer,
    pub(crate) vertex_candidates: Vec<AssignVertex>,
    pub(crate) propagator_candidates: Vec<AssignPropagator>,
    pub(crate) vertex_classification: VertexClassification,
    pub(crate) remaining_coupling_orders: Option<HashMap<String, usize>>,
}

impl<'a> AssignWorkspace<'a> {
    pub(crate) fn new(
        topology: &'a Topology,
        model: Arc<Model>,
        selector: &'a DiagramSelector,
        incoming_particles: &'a Vec<usize>,
        outgoing_particles: &'a Vec<usize>,
    ) -> Self {
        let remaining_coupling_orders = selector.get_max_coupling_orders();
        let vertex_candidates = topology
            .nodes_iter()
            .enumerate()
            .map(|(index, node)| {
                AssignVertex::new(
                    node.degree,
                    topology
                        .edges_iter()
                        .enumerate()
                        .filter_map(|(i, edge)| {
                            if edge.connected_nodes[0] == index || edge.connected_nodes[1] == index {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect_vec(),
                )
            })
            .collect_vec();
        let n_edges = topology.n_edges();
        return Self {
            topology,
            model,
            momentum_labels: Arc::new(topology.momentum_labels.clone()),
            selector,
            incoming_particles,
            outgoing_particles,
            diagram_buffer: DiagramContainer::new(None, &topology.momentum_labels),
            vertex_candidates,
            propagator_candidates: vec![AssignPropagator::new(); n_edges],
            vertex_classification: VertexClassification::from(topology),
            remaining_coupling_orders,
        };
    }

    pub fn assign(&mut self) -> DiagramContainer {
        let n_in = self.incoming_particles.len();
        let n_out = self.outgoing_particles.len();
        for i in 0..n_in {
            self.vertex_candidates[i].remaining_legs = 0;
            let edge = self.vertex_candidates[i].edges[0];
            if i == self.topology.edges[i].connected_nodes[0] {
                self.vertex_candidates[self.topology.edges[i].connected_nodes[1]].remaining_legs -= 1;
            } else {
                self.vertex_candidates[self.topology.edges[i].connected_nodes[0]].remaining_legs -= 1;
            }
            self.propagator_candidates[edge].particle = Some(self.incoming_particles[i]);
        }
        for i in 0..n_out {
            self.vertex_candidates[n_in + i].remaining_legs = 0;
            let edge = self.vertex_candidates[n_in + i].edges[0];
            if i == self.topology.edges[i].connected_nodes[0] {
                self.vertex_candidates[self.topology.edges[n_in + i].connected_nodes[1]].remaining_legs -= 1;
            } else {
                self.vertex_candidates[self.topology.edges[n_in + i].connected_nodes[0]].remaining_legs -= 1;
            }
            self.propagator_candidates[edge].particle = Some(self.outgoing_particles[i]);
        }
        for index in 0..self.vertex_candidates.len() {
            if self.vertex_candidates[index].degree == 1 {
                continue;
            }
            let connected_particles = self.get_connected_particles(index);
            let mut candidates = self
                .model
                .vertices_iter()
                .enumerate()
                .filter_map(|(i, interaction)| {
                    if interaction.degree() == self.vertex_candidates[index].degree {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect_vec();
            if !connected_particles.is_empty() {
                candidates.retain(|candidate| {
                    connected_particles
                        .iter()
                        .counts()
                        .into_iter()
                        .all(|(particle_index, count)| {
                            count <= self.model.vertex(*candidate).particle_count(particle_index)
                        })
                })
            }
            if candidates.is_empty() {
                return DiagramContainer::new(None, &self.topology.momentum_labels);
            }
            self.vertex_candidates[index].candidates = candidates;
        }
        self.select_vertex();
        let container = std::mem::replace(
            &mut self.diagram_buffer,
            DiagramContainer::new(None, &self.topology.momentum_labels),
        );
        return container;
    }

    fn select_vertex(&mut self) {
        let next_vertex = self
            .vertex_candidates
            .iter()
            .position_min_by_key(|candidate| {
                if candidate.remaining_legs == 0 {
                    usize::MAX
                } else {
                    candidate.remaining_legs
                }
            })
            .unwrap();

        if self.vertex_candidates[next_vertex].remaining_legs == 0 {
            if let Some(v_last) = self.vertex_candidates.iter().position(|c| c.candidates.len() > 1) {
                let candidates = self.vertex_candidates[v_last].candidates.clone();
                for c in candidates.iter() {
                    self.vertex_candidates[v_last].candidates = vec![*c];
                    if let Some(vertex_symmetry) = self.is_representative() {
                        let diagram = Diagram::from(self, vertex_symmetry);
                        if self
                            .selector
                            .select(self.model.clone(), self.momentum_labels.clone(), &diagram)
                        {
                            self.diagram_buffer.inner_ref_mut().push(diagram);
                        }
                    }
                }
                self.vertex_candidates[v_last].candidates = candidates;
            } else if let Some(vertex_symmetry) = self.is_representative() {
                let diagram = Diagram::from(self, vertex_symmetry);
                if self
                    .selector
                    .select(self.model.clone(), self.momentum_labels.clone(), &diagram)
                {
                    self.diagram_buffer.inner_ref_mut().push(diagram);
                }
            }
            return;
        }
        self.select_leg(next_vertex);
    }

    fn select_leg(&mut self, vertex: usize) {
        let next_leg = self.vertex_candidates[vertex]
            .edges
            .iter()
            .cloned()
            .enumerate()
            .find_map(|(index, edge)| {
                if self.propagator_candidates[edge].particle.is_none() {
                    Some(index)
                } else {
                    None
                }
            });
        match next_leg {
            None => {
                self.assign_vertex(vertex);
                return;
            }
            Some(next_leg) => {
                let current_vertex_candidates = self.vertex_candidates.clone();
                let current_propagator_candidates = self.propagator_candidates.clone();
                let current_vertex_classification = self.vertex_classification.clone();
                for particle in self.possible_particles(vertex, next_leg) {
                    if self.assign_particle(vertex, next_leg, particle) {
                        let other_vertex = if vertex
                            == self
                                .topology
                                .get_edge(self.vertex_candidates[vertex].edges[next_leg])
                                .connected_nodes[0]
                        {
                            self.topology
                                .get_edge(self.vertex_candidates[vertex].edges[next_leg])
                                .connected_nodes[1]
                        } else {
                            self.topology
                                .get_edge(self.vertex_candidates[vertex].edges[next_leg])
                                .connected_nodes[0]
                        };
                        self.vertex_candidates[vertex].remaining_legs -= 1;
                        self.vertex_candidates[other_vertex].remaining_legs -= 1;
                        self.select_leg(vertex);
                        self.vertex_candidates[vertex].remaining_legs += 1;
                        self.vertex_candidates[other_vertex].remaining_legs += 1;
                    }
                    self.vertex_candidates = current_vertex_candidates.clone();
                    self.propagator_candidates = current_propagator_candidates.clone();
                    self.vertex_classification = current_vertex_classification.clone();
                }
            }
        }
    }

    fn assign_vertex(&mut self, vertex: usize) {
        if self.vertex_candidates[vertex].candidates.len() == 1 {
            self.update_coupling_orders(self.vertex_candidates[vertex].candidates[0]);
            self.select_vertex();
            self.restore_coupling_orders(self.vertex_candidates[vertex].candidates[0]);
        } else {
            let current_vertex_candidates = self.vertex_candidates.clone();
            let current_propagator_candidates = self.propagator_candidates.clone();
            let candidates = self.vertex_candidates[vertex].candidates.clone();
            for candidate in candidates {
                self.vertex_candidates[vertex].candidates = vec![candidate];
                self.update_coupling_orders(self.vertex_candidates[vertex].candidates[0]);
                self.select_vertex();
                self.restore_coupling_orders(self.vertex_candidates[vertex].candidates[0]);
                self.vertex_candidates = current_vertex_candidates.clone();
                self.propagator_candidates = current_propagator_candidates.clone();
            }
        }
    }

    fn assign_particle(&mut self, vertex: usize, leg: usize, particle: usize) -> bool {
        let edge = self.vertex_candidates[vertex].edges[leg];
        let connected_vertex = if self.topology.get_edge(edge).connected_nodes[0] == vertex {
            self.topology.get_edge(edge).connected_nodes[1]
        } else {
            self.topology.get_edge(edge).connected_nodes[0]
        };
        self.propagator_candidates[edge].particle = Some(particle);
        self.update_vertex_candidates(vertex);
        self.update_vertex_candidates(connected_vertex);
        // Only check the connected vertex, the initial vertex is already checked when the possible particles
        // are determined
        if self.vertex_candidates[connected_vertex].candidates.is_empty() {
            return false;
        }
        return true;
    }

    fn possible_particles(&self, vertex: usize, leg: usize) -> Vec<usize> {
        let mut particles = Vec::new();
        let connected_particles = self.get_connected_particles(vertex);
        let edge = self.topology.get_edge(self.vertex_candidates[vertex].edges[leg]);
        let connected_vertex = if edge.connected_nodes[0] == vertex {
            edge.connected_nodes[1]
        } else {
            edge.connected_nodes[0]
        };
        for interaction_candidate in self.vertex_candidates[vertex].candidates.iter() {
            if !self
                .model
                .check_coupling_orders(*interaction_candidate, &self.remaining_coupling_orders)
            {
                continue;
            }
            for candidate_particle_str in self.model.vertex(*interaction_candidate).particles_iter() {
                let mut particle_index = self.model.get_particle_index(candidate_particle_str).unwrap();
                // If self-loop, only consider particles, not anti-particles (the anti-particles assigned here are
                // always inverted at the end of this function)
                if vertex == connected_vertex && !self.model.get_particle(particle_index).is_anti() {
                    particle_index = self.model.get_anti_index(particle_index)
                }
                if !particles.contains(&particle_index)
                    // Check for remaining open connections of `particle`
                    && (connected_particles.iter().filter(|i| **i == particle_index).count()
                        < self.model.vertex(*interaction_candidate).particle_count(&particle_index))
                    // Check if this assignment violates ordering of the edges into the connected class
                    && self.check_propagator_ordering(vertex, leg, particle_index)
                {
                    particles.push(particle_index);
                }
            }
        }
        // If looking at the propagator in the wrong direction, charge conjugate all candidates
        if edge.connected_nodes[0] == vertex {
            particles = particles
                .into_iter()
                .map(|p| self.model.get_anti_index(p))
                .collect_vec();
        }
        return particles;
    }

    /// Check if assigning `particle` to `leg` of `vertex` violates the ordering condition. The function returns `true`
    /// if all propagators are ordered correctly. All propagators between the given vertices are required to be maximal
    /// in lexicographical ordering.
    fn check_propagator_ordering(&self, vertex: usize, leg: usize, particle: usize) -> bool {
        let edge = self.vertex_candidates[vertex].edges[leg];
        // Assign `v`, `w` such, that `v --edge--> w`
        let v = self.topology.edges[edge].connected_nodes[0];
        let w = self.topology.edges[edge].connected_nodes[1];
        let ref_leg = self.vertex_candidates[v].edges.iter().position(|e| *e == edge).unwrap();
        let p = if self.topology.edges[edge].connected_nodes[0] == vertex {
            self.model.get_anti_index(particle)
        } else {
            particle
        };
        // Check ordering of edges between `v` and `w`
        if self.vertex_candidates[v]
            .edges
            .iter()
            .enumerate()
            .any(|(other_leg, cmp_edge)| {
                if self.propagator_candidates[*cmp_edge].particle.is_some()
                    && self.topology.edges[*cmp_edge].connected_nodes == [v, w]
                {
                    let other_particle = self.propagator_candidates[*cmp_edge].particle.unwrap();
                    (ref_leg < other_leg && p < other_particle) || (ref_leg > other_leg && p > other_particle)
                } else {
                    false
                }
            })
        {
            return false;
        }
        return true;
    }

    fn update_vertex_candidates(&mut self, vertex: usize) {
        let connected_particles = self.get_connected_particles(vertex);
        let vertex_candidates = &mut self.vertex_candidates[vertex];
        vertex_candidates.candidates.retain(|candidate| {
            if !self
                .model
                .check_coupling_orders(*candidate, &self.remaining_coupling_orders)
            {
                return false;
            }
            let mut counts = self.model.vertex(*candidate).particle_counts().clone();
            for particle in connected_particles.iter() {
                if let Some(c) = counts.get_mut(particle)
                    && *c > 0
                {
                    *c -= 1;
                } else {
                    return false;
                }
            }
            return true;
        });
    }

    fn update_coupling_orders(&mut self, vertex_id: usize) {
        let vertex = self.model.vertex(vertex_id);
        if let Some(remaining_powers) = self.remaining_coupling_orders.as_mut() {
            for (coupling, power) in vertex.coupling_orders() {
                if let Some(remaining_power) = remaining_powers.get_mut(coupling) {
                    *remaining_power -= power;
                } else {
                    continue;
                }
            }
        } else {
            return;
        }
    }

    fn restore_coupling_orders(&mut self, vertex_id: usize) {
        let vertex = self.model.vertex(vertex_id);
        if let Some(remaining_powers) = self.remaining_coupling_orders.as_mut() {
            for (coupling, power) in vertex.coupling_orders() {
                if let Some(remaining_power) = remaining_powers.get_mut(coupling) {
                    *remaining_power += power;
                } else {
                    continue;
                }
            }
        } else {
            return;
        }
    }

    fn get_connected_particles(&self, vertex: usize) -> Vec<usize> {
        let mut particles = vec![];
        for edge in self.vertex_candidates[vertex].edges.iter() {
            if self.propagator_candidates[*edge].particle.is_some() {
                if vertex == self.topology.get_edge(*edge).connected_nodes[0] {
                    particles.push(
                        self.model
                            .get_anti_index(self.propagator_candidates[*edge].particle.unwrap()),
                    );
                    // Self-loop -> the respective antiparticle is also connected to the vertex
                    if vertex == self.topology.get_edge(*edge).connected_nodes[1] {
                        particles.push(self.propagator_candidates[*edge].particle.unwrap());
                    }
                } else {
                    particles.push(self.propagator_candidates[*edge].particle.unwrap());
                    // Self-loop -> the respective antiparticle is also connected to the vertex
                    if vertex == self.topology.get_edge(*edge).connected_nodes[0] {
                        particles.push(
                            self.model
                                .get_anti_index(self.propagator_candidates[*edge].particle.unwrap()),
                        );
                    }
                }
            }
        }
        return particles;
    }

    fn is_representative(&self) -> Option<usize> {
        return generate_permutations(&self.vertex_classification.get_class_sizes())
            .fold_while(Some(0usize), |acc, permutation| {
                match self.cmp_permutation(&permutation) {
                    Ordering::Equal => FoldWhile::Continue(Some(acc.unwrap() + 1)),
                    Ordering::Greater => FoldWhile::Continue(Some(acc.unwrap())),
                    _ => FoldWhile::Done(None),
                }
            })
            .into_inner();
    }

    /// Check if the current diagram is larger in lexicographical ordering of the vertices and propagators
    /// than all possible permutations
    fn cmp_permutation(&self, perm: &[usize]) -> Ordering {
        let mut result = None;
        for (i, vertex) in self.vertex_candidates.iter().enumerate() {
            if vertex.degree == 1 {
                continue;
            }
            if result.is_none() {
                match vertex.candidates[0].cmp(&self.vertex_candidates[perm[i] - 1].candidates[0]) {
                    Ordering::Equal => {}
                    x => {
                        result = Some(x);
                    }
                }
            }
            for connected_vertex in self.topology.adjacent_nodes(i).iter() {
                if self.vertex_candidates[*connected_vertex].degree == 1
                    || (i == perm[i] - 1 && *connected_vertex == perm[*connected_vertex] - 1)
                {
                    // Current nodes not affected by permutation
                    continue;
                }
                if result.is_none() {
                    let ref_particle_ids = vertex
                        .edges
                        .iter()
                        .filter(|edge| {
                            self.topology.get_edge(**edge).connected_nodes == [i, *connected_vertex]
                                || self.topology.get_edge(**edge).connected_nodes == [*connected_vertex, i]
                        })
                        .map(|edge| self.propagator_candidates[*edge].particle.unwrap())
                        .sorted_unstable()
                        .collect_vec();
                    // Direction of the edges between the nodes is inverted by the permutation
                    let invert = (i < *connected_vertex && perm[i] > perm[*connected_vertex])
                        || (i > *connected_vertex && perm[i] < perm[*connected_vertex]);
                    let permuted_particle_ids = self.vertex_candidates[perm[i] - 1]
                        .edges
                        .iter()
                        .filter(|edge| {
                            self.topology.get_edge(**edge).connected_nodes == [perm[i] - 1, perm[*connected_vertex] - 1]
                                || self.topology.get_edge(**edge).connected_nodes
                                    == [perm[*connected_vertex] - 1, perm[i] - 1]
                        })
                        .map(|edge| {
                            if invert {
                                self.model
                                    .get_anti_index(self.propagator_candidates[*edge].particle.unwrap())
                            } else {
                                self.propagator_candidates[*edge].particle.unwrap()
                            }
                        })
                        .sorted_unstable()
                        .collect_vec();
                    match ref_particle_ids.len().cmp(&permuted_particle_ids.len()) {
                        Ordering::Equal => (),
                        Ordering::Greater => return Ordering::Greater,
                        x => result = Some(x),
                    }
                    for (ref_p, perm_p) in ref_particle_ids.into_iter().zip(permuted_particle_ids) {
                        match ref_p.cmp(&perm_p) {
                            Ordering::Equal => (),
                            x => {
                                result = Some(x);
                            }
                        }
                    }
                } else {
                    // Ordering is already defined if result is Some, therefore only need to check for validity of permutation
                    let ref_particle_len = vertex
                        .edges
                        .iter()
                        .filter(|edge| {
                            self.topology.get_edge(**edge).connected_nodes == [i, *connected_vertex]
                                || self.topology.get_edge(**edge).connected_nodes == [*connected_vertex, i]
                        })
                        .count();
                    let permuted_particle_len = self.vertex_candidates[perm[i] - 1]
                        .edges
                        .iter()
                        .filter(|edge| {
                            self.topology.get_edge(**edge).connected_nodes == [perm[i] - 1, perm[*connected_vertex] - 1]
                                || self.topology.get_edge(**edge).connected_nodes
                                    == [perm[*connected_vertex] - 1, perm[i] - 1]
                        })
                        .count();
                    match ref_particle_len.cmp(&permuted_particle_len) {
                        Ordering::Equal => (),
                        x => return x,
                    }
                }
            }
        }
        if let Some(result) = result {
            return result;
        }
        return Ordering::Equal;
    }

    fn trace_fermi_line(&self, prop: usize, visited: &mut [bool]) -> (usize, usize) {
        let initial_vertex = self.topology.edges[prop].connected_nodes[0];
        let mut to_visit: Vec<usize> = Vec::new();
        to_visit.push(self.topology.edges[prop].connected_nodes[1]);
        visited[prop] = true;
        while let Some(current) = to_visit.pop() {
            for edge in self.vertex_candidates[current].edges.iter() {
                if visited[*edge] {
                    continue;
                }
                visited[*edge] = true;
                if !self
                    .model
                    .get_particle(*self.propagator_candidates[*edge].particle.as_ref().unwrap())
                    .is_fermi()
                {
                    continue;
                }
                let connected_vertex = if self.topology.edges[*edge].connected_nodes[0] == current {
                    self.topology.edges[*edge].connected_nodes[1]
                } else {
                    self.topology.edges[*edge].connected_nodes[0]
                };
                if connected_vertex == initial_vertex || self.topology.nodes[connected_vertex].degree == 1 {
                    return (initial_vertex, connected_vertex);
                }
                to_visit.push(connected_vertex);
            }
        }
        return (initial_vertex, initial_vertex);
    }

    pub(crate) fn calculate_sign(&self) -> i8 {
        let mut visited = vec![false; self.propagator_candidates.len()];
        let mut external_fermions = Vec::new();
        for external_node in self.vertex_candidates.iter().take(self.topology.n_external) {
            let edge = external_node.edges[0];
            if visited[edge]
                || !self
                    .model
                    .get_particle(self.propagator_candidates[edge].particle.unwrap())
                    .is_fermi()
            {
                continue;
            }
            let (initial_vertex, final_vertex) = self.trace_fermi_line(edge, &mut visited);
            external_fermions.push(initial_vertex);
            external_fermions.push(final_vertex);
        }
        let mut n_ext_swap: usize = 0;
        for i in 0..external_fermions.len() {
            for j in i + 1..external_fermions.len() {
                if external_fermions[i] > external_fermions[j] {
                    n_ext_swap += 1;
                }
            }
        }
        let mut fermi_loops: usize = 0;
        for edge in 0..self.propagator_candidates.len() {
            if visited[edge]
                || !self
                    .model
                    .get_particle(self.propagator_candidates[edge].particle.unwrap())
                    .is_fermi()
            {
                continue;
            }
            let _ = self.trace_fermi_line(edge, &mut visited);
            fermi_loops += 1;
        }
        return if (n_ext_swap + fermi_loops) % 2 == 0 { 1 } else { -1 };
    }
}

#[cfg(test)]
mod tests {
    use crate::diagram::filter::DiagramSelector;
    use crate::diagram::workspace::AssignWorkspace;
    use crate::model::{Model, TopologyModel};
    use crate::topology::filter::TopologySelector;
    use crate::topology::{Edge, Node, Topology, TopologyGenerator, components::NodeClassification};
    use std::path::PathBuf;
    use std::sync::Arc;
    use test_log::test;

    #[test]
    fn assign_qcd_gluon_4point_tree_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_UFO")).unwrap();
        let incoming = vec![model.get_particle_index("G").unwrap().clone(); 2];
        let outgoing = incoming.clone();
        let topology = Topology {
            n_external: 4,
            n_loops: 0,
            nodes: vec![
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 4,
                    adjacent_nodes: vec![0, 1, 2, 3],
                },
            ],
            edges: vec![
                Edge {
                    connected_nodes: [0, 4],
                    momenta: Some(vec![1, 0, 0, 0]),
                },
                Edge {
                    connected_nodes: [1, 4],
                    momenta: Some(vec![0, 1, 0, 0]),
                },
                Edge {
                    connected_nodes: [2, 4],
                    momenta: Some(vec![0, 0, 1, 0]),
                },
                Edge {
                    connected_nodes: [3, 4],
                    momenta: Some(vec![0, 0, 0, 1]),
                },
            ],
            node_symmetry: 1,
            edge_symmetry: 1,
            momentum_labels: vec![
                String::from("p1"),
                String::from("p2"),
                String::from("p3"),
                String::from("p4"),
            ],
            bridges: vec![],
            node_classification: NodeClassification {
                boundaries: vec![0, 1, 2, 3, 4, 5],
                matrix: vec![
                    vec![0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1],
                    vec![1, 1, 1, 1, 0],
                ],
            },
        };
        let selector = DiagramSelector::default();
        let mut workspace = AssignWorkspace::new(&topology, Arc::new(model), &selector, &incoming, &outgoing);
        let container = workspace.assign();
        assert_eq!(container.len(), 1);
    }

    #[test]
    fn assign_qcd_gluon_s_channel_tree_test() {
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_UFO")).unwrap();
        let incoming = vec![model.get_particle_index("G").unwrap().clone(); 2];
        let outgoing = incoming.clone();
        let topology = Topology {
            n_external: 4,
            n_loops: 0,
            nodes: vec![
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 3,
                    adjacent_nodes: vec![0, 1, 5],
                },
                Node {
                    degree: 3,
                    adjacent_nodes: vec![2, 3, 4],
                },
            ],
            edges: vec![
                Edge {
                    connected_nodes: [0, 4],
                    momenta: Some(vec![1, 0, 0, 0]),
                },
                Edge {
                    connected_nodes: [1, 4],
                    momenta: Some(vec![0, 1, 0, 0]),
                },
                Edge {
                    connected_nodes: [2, 5],
                    momenta: Some(vec![0, 0, 1, 0]),
                },
                Edge {
                    connected_nodes: [3, 5],
                    momenta: Some(vec![0, 0, 0, 1]),
                },
                Edge {
                    connected_nodes: [4, 5],
                    momenta: Some(vec![1, 1, 0, 0]),
                },
            ],
            node_symmetry: 1,
            edge_symmetry: 1,
            momentum_labels: vec![
                String::from("p1"),
                String::from("p2"),
                String::from("p3"),
                String::from("p4"),
            ],
            bridges: vec![(4, 5)],
            node_classification: NodeClassification {
                boundaries: vec![0, 1, 2, 3, 4, 5, 6],
                matrix: vec![
                    vec![0, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 0, 1],
                    vec![1, 1, 0, 0, 0, 1],
                    vec![0, 0, 1, 1, 1, 0],
                ],
            },
        };
        let selector = DiagramSelector::default();
        let mut workspace = AssignWorkspace::new(&topology, Arc::new(model), &selector, &incoming, &outgoing);
        let container = workspace.assign();
        assert_eq!(container.len(), 1);
    }

    #[test]
    pub fn workspace_qcd_g_prop_4loop() {
        let mut topo_selector = TopologySelector::new();
        topo_selector.select_opi_components(1);
        let topos = TopologyGenerator::new(2, 4, TopologyModel::from(vec![3, 4]), Some(topo_selector)).generate();
        let topo = topos[1004].clone();
        let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_U_UFO")).unwrap();
        let selector = DiagramSelector::default();
        let particles_in = vec![model.get_particle_index("G").unwrap().clone()];
        let particle_out = vec![model.get_particle_index("G").unwrap().clone()];
        let mut workspace = AssignWorkspace::new(&topo, Arc::new(model), &selector, &particles_in, &particle_out);
        let diagrams = workspace.assign();
        assert_eq!(diagrams.len(), 13);
    }
}
