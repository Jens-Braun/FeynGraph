//! Public interface objects for the internal representation of diagrams.

use crate::{
    diagram::{Diagram, Leg, Propagator, Vertex},
    model::{InteractionVertex, Model, Particle},
};
use either::Either;
use itertools::Itertools;
use std::fmt::Write;

/// Public interface object of a [`Diagram`].
pub struct DiagramView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) diagram: &'a Diagram,
    pub(crate) momentum_labels: &'a Vec<String>,
}

impl<'a> DiagramView<'a> {
    pub(crate) fn new(model: &'a Model, diagram: &'a Diagram, momentum_labels: &'a Vec<String>) -> Self {
        return Self {
            model,
            diagram,
            momentum_labels,
        };
    }

    /// Get an iterator over the incoming legs.
    pub fn incoming(&self) -> impl Iterator<Item = LegView<'_>> {
        return self.diagram.incoming_legs.iter().enumerate().map(|(i, p)| LegView {
            model: self.model,
            diagram: self,
            leg: p,
            leg_index: i,
            invert_particle: false,
        });
    }

    /// Get an iterator over the outgoing legs.
    pub fn outgoing(&self) -> impl Iterator<Item = LegView<'_>> {
        return self.diagram.outgoing_legs.iter().enumerate().map(|(i, p)| LegView {
            model: self.model,
            diagram: self,
            leg: p,
            leg_index: i + self.diagram.incoming_legs.len(),
            invert_particle: true,
        });
    }

    /// Get an iterator over the internal propagators.
    pub fn propagators(&self) -> impl Iterator<Item = PropagatorView<'_>> {
        return self
            .diagram
            .propagators
            .iter()
            .enumerate()
            .map(|(i, p)| PropagatorView {
                model: self.model,
                diagram: self,
                propagator: p,
                index: i,
                invert: false,
            });
    }

    /// Get the `index`-th internal propagator.
    pub fn propagator(&self, index: usize) -> PropagatorView<'_> {
        return PropagatorView {
            model: self.model,
            diagram: self,
            propagator: &self.diagram.propagators[index],
            index,
            invert: false,
        };
    }

    /// Get the `index`-th internal vertex.
    pub fn vertex(&self, index: usize) -> VertexView<'_> {
        return VertexView {
            model: self.model,
            diagram: self,
            vertex: &self.diagram.vertices[index],
            index,
        };
    }

    /// Get an iterator over the internal vertices.
    pub fn vertices(&self) -> impl Iterator<Item = VertexView<'_>> {
        return self.diagram.vertices.iter().enumerate().map(|(i, v)| VertexView {
            model: self.model,
            diagram: self,
            vertex: v,
            index: i,
        });
    }

    /// Get an iterator over the vertices belonging to the `index`-th loop.
    pub fn loop_vertices(&self, index: usize) -> impl Iterator<Item = VertexView<'_>> {
        let loop_index = self.n_ext() + index;
        return self.diagram.vertices.iter().enumerate().filter_map(move |(i, v)| {
            if self.diagram.vertices[index]
                .propagators
                .iter()
                .any(|j| *j >= 0 && self.diagram.propagators[*j as usize].momentum[loop_index] != 0)
            {
                Some(VertexView {
                    model: self.model,
                    diagram: self,
                    vertex: v,
                    index: i,
                })
            } else {
                None
            }
        });
    }

    /// Get an iterator over the propagators belonging to the `index`-th loop.
    pub fn chord(&self, index: usize) -> impl Iterator<Item = PropagatorView<'_>> {
        let loop_index = self.n_ext() + index;
        return self
            .diagram
            .propagators
            .iter()
            .enumerate()
            .filter_map(move |(i, prop)| {
                if prop.momentum[loop_index] != 0 {
                    Some(self.propagator(i))
                } else {
                    None
                }
            });
    }

    /// Get an iterator over the bridge propagators of the diagram.
    pub fn bridges(&self) -> impl Iterator<Item = PropagatorView<'_>> {
        return self.diagram.bridges.iter().map(|i| self.propagator(*i));
    }

    /// Get the number of external legs.
    pub fn n_ext(&self) -> usize {
        return self.diagram.incoming_legs.len() + self.diagram.outgoing_legs.len();
    }

    /// Get the diagram's symmetry factor.
    pub fn symmetry_factor(&self) -> usize {
        return self.diagram.vertex_symmetry * self.diagram.propagator_symmetry;
    }

    /// Get the diagram's relative sign.
    pub fn sign(&self) -> i8 {
        return self.diagram.sign;
    }

    fn trace_fermi_line(
        &self,
        vertex: VertexView,
        ray: usize,
        visited_props: &mut [bool],
        visited_legs: &mut [bool],
    ) -> usize {
        let initial_vertex = vertex.index;
        let mut to_visit: Vec<VertexView> = vec![vertex];
        let mut in_ray = ray;
        while let Some(current) = to_visit.pop() {
            let out_ray = current.interaction().spin_map[in_ray] as usize;
            let out_prop = current.propagators_ordered().nth(out_ray).unwrap();
            match out_prop {
                Either::Left(leg) => {
                    if visited_legs[leg.leg_index] {
                        continue;
                    }
                    visited_legs[leg.leg_index] = true;
                    return leg.leg_index;
                }
                Either::Right(prop) => {
                    if visited_props[prop.index] {
                        continue;
                    }
                    visited_props[prop.index] = true;
                    if prop.propagator.vertices[0] == current.index {
                        in_ray = prop.ray_index_ordered(1);
                        to_visit.push(self.vertex(prop.propagator.vertices[1]));
                    } else {
                        in_ray = prop.ray_index_ordered(0);
                        to_visit.push(self.vertex(prop.propagator.vertices[0]));
                    }
                }
            }
        }
        return initial_vertex;
    }

    pub(crate) fn calculate_sign(&self) -> i8 {
        let mut visited_legs = vec![false; self.n_ext()];
        let mut visited_props = vec![false; self.diagram.propagators.len()];
        let mut external_fermions = Vec::new();
        for leg in self.incoming().chain(self.outgoing()) {
            let vertex = leg.vertex();
            let ray = leg.ray_index_ordered();
            if visited_legs[leg.leg_index] || !leg.particle().is_fermi() {
                continue;
            }
            visited_legs[leg.leg_index] = true;
            let final_leg = self.trace_fermi_line(vertex, ray, &mut visited_props, &mut visited_legs);
            if leg.leg_index < final_leg {
                external_fermions.push(leg.leg_index);
                external_fermions.push(final_leg);
            } else {
                external_fermions.push(final_leg);
                external_fermions.push(leg.leg_index);
            }
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
        for prop in self.propagators() {
            if visited_props[prop.index] || !prop.particle().is_fermi() {
                continue;
            }
            if prop.propagator.vertices[0] == prop.propagator.vertices[1] {
                fermi_loops += 1;
                visited_props[prop.index] = true;
                continue;
            }
            let vertex = prop.vertex(1);
            let ray = prop.ray_index_ordered(1);
            let _ = self.trace_fermi_line(vertex, ray, &mut visited_props, &mut visited_legs);
            fermi_loops += 1;
        }
        return if (n_ext_swap + fermi_loops) % 2 == 0 { 1 } else { -1 };
    }
}

impl std::fmt::Display for DiagramView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Diagram {{")?;
        write!(f, "    Process: ")?;
        for incoming in self.incoming() {
            write!(f, "{} ", incoming.particle().name())?;
        }
        write!(f, "-> ")?;
        for outgoing in self.outgoing() {
            write!(f, "{} ", outgoing.particle().name())?;
        }
        writeln!(f)?;
        write!(f, "    Vertices: [ ")?;
        for vertex in self.vertices() {
            write!(f, "{} ", vertex)?;
        }
        writeln!(f, "]")?;
        writeln!(f, "    Legs: [")?;
        for leg in self.incoming() {
            writeln!(f, "        {}", leg)?;
        }
        for leg in self.outgoing() {
            writeln!(f, "        {}", leg)?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "    Propagators: [")?;
        for propagator in self.propagators() {
            writeln!(f, "        {}", propagator)?;
        }
        writeln!(f, "    ]")?;
        writeln!(
            f,
            "    SymmetryFactor: 1/{}",
            self.diagram.vertex_symmetry * self.diagram.propagator_symmetry
        )?;
        writeln!(f, "    Sign: {}", if self.diagram.sign == 1 { "+" } else { "-" })?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

/// Public interface of an external [`Leg`].
#[derive(Clone)]
pub struct LegView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) diagram: &'a DiagramView<'a>,
    pub(crate) leg: &'a Leg,
    pub(crate) leg_index: usize,
    pub(crate) invert_particle: bool,
}

impl LegView<'_> {
    /// Get the vertex the leg is attached to.
    pub fn vertex(&self) -> VertexView<'_> {
        return self.diagram.vertex(self.leg.vertex);
    }

    /// Get the particle assigned to the leg
    pub fn particle(&self) -> &Particle {
        return if self.invert_particle {
            self.model.get_anti(self.leg.particle)
        } else {
            self.model.get_particle(self.leg.particle)
        };
    }

    /// Get the external leg's ray index, i.e. the index of the leg of the vertex to which the external leg is
    /// connected to (_from the vertex perspective_).
    pub fn ray_index(&self) -> usize {
        return self.diagram.diagram.vertices[self.leg.vertex]
            .propagators
            .iter()
            .position(|p| (*p + self.diagram.n_ext() as isize) as usize == self.leg_index)
            .unwrap();
    }

    /// Get the external leg's ray index, i.e. the index of the leg of the vertex to which the external leg is
    /// connected to (_from the vertex perspective_). The ray index is given with respect to the propagators ordered
    /// as in the interaction vertex.
    pub fn ray_index_ordered(&self) -> usize {
        return self
            .vertex()
            .propagators_ordered()
            .position(|p| p.left().map(|l| l.leg_index) == Some(self.leg_index))
            .unwrap();
    }

    /// Get the string-formatted momentum flowing through the leg.
    pub fn momentum_str(&self) -> String {
        let mut result = String::with_capacity(5 * self.diagram.momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.leg.momentum.iter().enumerate() {
            if *coefficient == 0 {
                continue;
            }
            match *coefficient {
                1 => {
                    if !first {
                        write!(&mut result, "+").unwrap();
                    } else {
                        first = false;
                    }
                    write!(&mut result, "{}", self.diagram.momentum_labels[i]).unwrap();
                }
                -1 => {
                    write!(&mut result, "-{}", self.diagram.momentum_labels[i]).unwrap();
                }
                x if x < 0 => write!(&mut result, "-{}*{}", x.abs(), self.diagram.momentum_labels[i]).unwrap(),
                x => write!(&mut result, "{}*{}", x, self.diagram.momentum_labels[i]).unwrap(),
            }
        }
        return result;
    }
}

impl std::fmt::Display for LegView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.leg_index >= self.diagram.diagram.incoming_legs.len() {
            write!(
                f,
                "{}[{} -> ], p = {},",
                self.particle().name(),
                self.leg.vertex,
                self.momentum_str()
            )?;
        } else {
            write!(
                f,
                "{}[-> {}], p = {},",
                self.particle().name(),
                self.leg.vertex,
                self.momentum_str()
            )?;
        }
        Ok(())
    }
}

/// Public interface of an internal [`Propagator`].
#[derive(Clone)]
pub struct PropagatorView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) diagram: &'a DiagramView<'a>,
    pub(crate) propagator: &'a Propagator,
    pub(crate) index: usize,
    pub(crate) invert: bool,
}

impl PropagatorView<'_> {
    /// Get an iterator over the vertices connected by the propagator.
    pub fn vertices(&self) -> impl Iterator<Item = VertexView<'_>> {
        return if self.invert {
            [
                self.diagram.vertex(self.propagator.vertices[1]),
                self.diagram.vertex(self.propagator.vertices[0]),
            ]
            .into_iter()
        } else {
            [
                self.diagram.vertex(self.propagator.vertices[0]),
                self.diagram.vertex(self.propagator.vertices[1]),
            ]
            .into_iter()
        };
    }

    /// Get the `index`-th vertex connected to the propagator.
    pub fn vertex(&self, index: usize) -> VertexView<'_> {
        let i = if self.invert { 1 - index } else { index };
        return self.diagram.vertex(self.propagator.vertices[i]);
    }

    /// Get the particle assigned to the propagator.
    pub fn particle(&self) -> &Particle {
        return if self.invert {
            self.model.get_anti(self.propagator.particle)
        } else {
            self.model.get_particle(self.propagator.particle)
        };
    }

    /// Get the propagators ray index with respect to the `index`-th vertex it is connected to, i.e. the index of the
    /// leg of the `index`-th vertex to which the propagator is connected to.
    pub fn ray_index(&self, index: usize) -> usize {
        let i = if self.invert { 1 - index } else { index };
        let pos = self.diagram.diagram.vertices[self.propagator.vertices[i]]
            .propagators
            .iter()
            .position(|p| *p == self.index as isize)
            .unwrap();
        return if self.propagator.vertices[0] == self.propagator.vertices[1] {
            // If the propagator is a self-loop, the vertex' propagator list contains `self.index` twice - at `pos` and at `pos+1`.
            // Since the direction of a self-loop is always ambiguous, it is fixed such that it always runs from the leg at `pos`
            // to the leg at `pos+1`.
            pos + i
        } else {
            pos
        };
    }

    /// Get the propagators ray index with respect to the `index`-th vertex it is connected to, i.e. the index of the
    /// leg of the `index`-th vertex to which the propagator is connected to. The ray index is given with respect to the
    /// propagators ordered as in the interaction vertex.
    pub fn ray_index_ordered(&self, index: usize) -> usize {
        let i = if self.invert { 1 - index } else { index };
        return self
            .vertex(i)
            .propagators_ordered()
            .position(|p| {
                if let either::Right(p) = p
                    && p.index == self.index
                {
                    if self.propagator.vertices[0] == self.propagator.vertices[1] {
                        if i == 1 {
                            return p.particle().is_anti();
                        } else {
                            return !p.particle().is_anti();
                        }
                    } else {
                        return true;
                    }
                } else {
                    false
                }
            })
            .unwrap();
    }

    /// Get the internal representation of the momentum flowing through the propagator.
    pub fn momentum(&self) -> Vec<i8> {
        return if self.invert {
            self.propagator.momentum.iter().map(|x| -*x).collect_vec()
        } else {
            self.propagator.momentum.clone()
        };
    }

    /// Get the string-formatted momentum flowing through the propagator.
    pub fn momentum_str(&self) -> String {
        let momentum_labels = self.diagram.momentum_labels;
        let mut result = String::with_capacity(5 * momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.propagator.momentum.iter().enumerate() {
            if *coefficient == 0 {
                continue;
            }
            let sign;
            if first {
                sign = "";
                first = false;
            } else {
                sign = "+";
            }
            match *coefficient * if self.invert { -1 } else { 1 } {
                1 => write!(&mut result, "{}{}", sign, momentum_labels[i]).unwrap(),
                -1 => write!(&mut result, "-{}", momentum_labels[i]).unwrap(),
                x if x < 0 => write!(&mut result, "-{}*{}", x.abs(), momentum_labels[i]).unwrap(),
                x => write!(&mut result, "{}{}*{}", sign, x, momentum_labels[i]).unwrap(),
            }
        }
        return result;
    }
}

impl std::fmt::Display for PropagatorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.invert {
            write!(
                f,
                "{}[{} -> {}], p = {},",
                self.particle().name(),
                self.propagator.vertices[1],
                self.propagator.vertices[0],
                self.momentum_str()
            )?;
        } else {
            write!(
                f,
                "{}[{} -> {}], p = {},",
                self.particle().name(),
                self.propagator.vertices[0],
                self.propagator.vertices[1],
                self.momentum_str()
            )?;
        }
        Ok(())
    }
}

/// Public interface of an internal [`Vertex`].
pub struct VertexView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) diagram: &'a DiagramView<'a>,
    pub(crate) vertex: &'a Vertex,
    pub(crate) index: usize,
}

impl VertexView<'_> {
    /// Get an iterator over the propagators connected to the vertex.
    pub fn propagators(&self) -> impl Iterator<Item = Either<LegView<'_>, PropagatorView<'_>>> {
        return self.vertex.propagators.iter().enumerate().map(|(i, prop)| {
            if *prop >= 0 {
                Either::Right(PropagatorView {
                    model: self.model,
                    diagram: self.diagram,
                    propagator: &self.diagram.diagram.propagators[*prop as usize],
                    index: *prop as usize,
                    invert: (self.diagram.diagram.propagators[*prop as usize].vertices[0] == self.index
                        && self.diagram.diagram.propagators[*prop as usize].vertices[1] != self.index)
                        || (i > 0 && self.vertex.propagators[i - 1] == self.vertex.propagators[i]),
                })
            } else {
                let index = (*prop + self.diagram.n_ext() as isize) as usize;
                let leg = if index < self.diagram.diagram.incoming_legs.len() {
                    &self.diagram.diagram.incoming_legs[index]
                } else {
                    &self.diagram.diagram.outgoing_legs[index - self.diagram.diagram.incoming_legs.len()]
                };
                Either::Left(LegView {
                    model: self.model,
                    diagram: self.diagram,
                    leg,
                    leg_index: index,
                    invert_particle: false,
                })
            }
        });
    }

    /// Get an iterator over the propagators connected to the vertex ordered like the particles in the interaction.
    pub fn propagators_ordered(&self) -> impl Iterator<Item = Either<LegView<'_>, PropagatorView<'_>>> {
        let views = self.propagators().collect_vec();
        let mut perm = Vec::with_capacity(views.len());
        let mut seen = vec![false; views.len()];
        for ref_particle in self.model.vertex(self.vertex.interaction).particles.iter() {
            for (i, part) in views
                .iter()
                .map(|view| either::for_both!(view, p => p.particle()))
                .enumerate()
            {
                if !seen[i] && part.name() == ref_particle {
                    perm.push(i);
                    seen[i] = true;
                } else {
                    continue;
                }
            }
        }
        return perm.into_iter().map(move |i| views[i].clone());
    }

    /// Get the interaction assigned to the vertex.
    pub fn interaction(&self) -> &InteractionVertex {
        return self.model.vertex(self.vertex.interaction);
    }

    /// Check whether the given particle names match the interaction of the vertex.
    pub fn match_particles<'q>(&self, query: impl IntoIterator<Item = &'q String>) -> bool {
        return self
            .model
            .vertex(self.vertex.interaction)
            .particles
            .iter()
            .sorted()
            .zip(query.into_iter().sorted())
            .all(|(part, query)| *part == *query);
    }
}

impl std::fmt::Display for VertexView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}[ {}: ",
            self.index,
            self.model.vertex(self.vertex.interaction).name
        )?;
        for p in self.model.vertex(self.vertex.interaction).particles.iter() {
            write!(f, "{} ", p)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}
