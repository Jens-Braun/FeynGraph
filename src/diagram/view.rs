use std::fmt::Write;
use itertools::Itertools;
use either::Either;
use crate::{
    diagram::{Diagram, Propagator, Vertex, Leg},
    model::{Model, Particle, InteractionVertex}
};

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
            momentum_labels
        }
    }

    /// Get an iterator over the incoming legs
    pub fn incoming(&self) -> impl Iterator<Item = LegView> {
        return self.diagram.incoming_legs.iter().enumerate().map(
            |(i, p)| LegView {
                model: self.model,
                diagram: self,
                leg: p,
                leg_index: i
            }
        );
    }

    /// Get an iterator over the outgoing legs
    pub fn outgoing(&self) -> impl Iterator<Item = LegView> {
        return self.diagram.outgoing_legs.iter().enumerate().map(
            |(i, p)| LegView {
                model: self.model,
                diagram: self,
                leg: p,
                leg_index: i + self.diagram.incoming_legs.len()
            }
        );
    }

    /// Get an iterator over the internal propagators
    pub fn propagators(&self) -> impl Iterator<Item = PropagatorView> {
        return self.diagram.propagators.iter().enumerate()
            .map(
                |(i, p)| PropagatorView {
                    model: self.model,
                    diagram: self,
                    propagator: p,
                    index: i
                }
            )
    }

    /// Get the `index`-th internal propagator
    pub fn propagator(&self, index: usize) -> PropagatorView {
        return PropagatorView {
            model: self.model,
            diagram: self,
            propagator: &self.diagram.propagators[index],
            index
        }
    }

    /// Get the `index`-th internal vertex
    pub fn vertex(&self, index: usize) -> VertexView {
        return VertexView {
            model: self.model,
            diagram: self,
            vertex: &self.diagram.vertices[index],
            index
        }
    }

    /// Get an iterator over the internal vertices
    pub fn vertices(&self) -> impl Iterator<Item = VertexView> {
        return self.diagram.vertices.iter().enumerate().map(
            |(i, v)| VertexView {
                model: self.model,
                diagram: self,
                vertex: v,
                index: i
            }
        )
    }

    /// Get an iterator over the vertices belonging to the `index`-th loop
    pub fn loop_vertices(&self, index: usize) -> impl Iterator<Item = VertexView> {
        let loop_index = self.n_ext() + index;
        return self.diagram.vertices.iter().enumerate().filter_map(
            move |(i, v)| if self.diagram.vertices[index].propagators.iter().any(
                |j| *j >= 0 && self.diagram.propagators[*j as usize].momentum[loop_index] != 0
            ) {
                Some(VertexView {
                    model: self.model,
                    diagram: self,
                    vertex: v,
                    index: i
                })
            } else {
                None
            }
        )
    }

    /// Get an iterator over the propagators belonging to the `index`-th loop
    pub fn chord(&self, index: usize) -> impl Iterator<Item = PropagatorView> {
        let loop_index = self.n_ext() + index;
        return self.diagram.propagators.iter().enumerate().filter_map(
            move |(i, prop)| if prop.momentum[loop_index] != 0 {
                Some(self.propagator(i))
            } else {
                None
            }
        );
    }

    /// Get an iterator over the bridge propagators of the diagram
    pub fn bridges(&self) -> impl Iterator<Item = PropagatorView> {
        return self.diagram.bridges.iter().map(
            |i| self.propagator(*i)
        );
    }

    /// Get the number of external legs
    pub fn n_ext(&self) -> usize {
        return self.diagram.incoming_legs.len() + self.diagram.outgoing_legs.len();
    }

    /// Get the diagram's symmetry factor
    pub fn symmetry_factor(&self) -> usize {
        return self.diagram.vertex_symmetry * self.diagram.propagator_symmetry;
    }

    /// Get the diagram's relative sign
    pub fn sign(&self) -> i8 {
        return self.diagram.sign;
    }
}

impl std::fmt::Display for DiagramView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Diagram {{")?;
        write!(f, "    Process: ")?;
        for incoming in self.incoming() {
            write!(f, "{} ", incoming.particle().get_name())?;
        }
        write!(f, "-> ")?;
        for outgoing in self.outgoing() {
            write!(f, "{} ", outgoing.particle().get_name())?;
        }
        writeln!(f, "")?;
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
        writeln!(f, "    SymmetryFactor: 1/{}", self.diagram.vertex_symmetry * self.diagram.propagator_symmetry)?;
        writeln!(f, "    Sign: {}", if self.diagram.sign == 1 {"+"} else {"-"})?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct LegView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) diagram: &'a DiagramView<'a>,
    pub(crate) leg: &'a Leg,
    pub(crate) leg_index: usize
}

impl<'a> LegView<'a> {
    /// Get the vertex the leg is attached to
    pub fn vertex(&self) -> VertexView {
        return self.diagram.vertex(self.leg.vertex);
    }

    /// Get the particle assigned to the leg
    pub fn particle(&self) -> &Particle {
        return self.model.get_particle(self.leg.particle);
    }

    /// Get the external leg's ray index, i.e. the index of the leg of the vertex to which the external leg is
    /// connected to (_from the vertex perspective_)
    pub fn ray_index(&self) -> usize {
        return self.diagram.diagram.vertices[self.leg.vertex]
            .propagators.iter().position(
            |p| (*p + self.diagram.n_ext() as isize) as usize == self.leg_index
        ).unwrap();
    }

    /// Get the string-formatted momentum flowing through the leg
    pub fn momentum_str(&self) -> String {
        let mut result = String::with_capacity(5*self.diagram.momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.leg.momentum.iter().enumerate() {
            if *coefficient == 0 { continue; }
            match *coefficient {
                1 => {
                    if !first {
                        write!(&mut result, "+").unwrap();
                    } else {
                        first = false;
                    }
                    write!(&mut result, "{}", self.diagram.momentum_labels[i]).unwrap();
                },
                -1 => {
                    write!(&mut result, "-{}", self.diagram.momentum_labels[i]).unwrap();
                },
                x if x < 0 => write!(&mut result, "-{}*{}", x.abs(), self.diagram.momentum_labels[i]).unwrap(),
                x => write!(&mut result, "{}*{}", x, self.diagram.momentum_labels[i]).unwrap()
            }
        }
        return result;
    }
}

impl std::fmt::Display for LegView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[{}], p = {},",
               self.particle().get_name(),
               self.leg.vertex,
               self.momentum_str()
        )?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct PropagatorView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) diagram: &'a DiagramView<'a>,
    pub(crate) propagator: &'a Propagator,
    pub(crate) index: usize,
}

impl<'a> PropagatorView<'a> {
    /// Get an iterator over the vertices connected by the propagator
    pub fn vertices(&self) -> impl Iterator<Item = VertexView> {
        return self.propagator.vertices.iter().map(
            |i| self.diagram.vertex(*i)
        )
    }

    /// Get the `index`-th vertex connected to the propagator
    pub fn vertex(&self, index: usize) -> VertexView {
        return self.diagram.vertex(self.propagator.vertices[index]);
    }

    /// Get the particle assigned to the propagator
    pub fn particle(&self) -> &Particle {
        return self.model.get_particle(self.propagator.particle)
    }

    /// Get the propagators ray index with respect to the `index`-th vertex it is connected to, i.e. the index of the
    /// leg of the `index`-th vertex to which the propagator is connected to
    pub fn ray_index(&self, index: usize) -> usize {
        return self.diagram.diagram.vertices[self.propagator.vertices[index]]
            .propagators.iter().position(|p| *p == self.index as isize).unwrap();
    }

    /// Get the internal representation of the momentum flowing through the propagator
    pub fn momentum(&self) -> &[i8] {
        return &self.propagator.momentum;
    }

    /// Get the string-formatted momentum flowing through the propagator
    pub fn momentum_str(&self) -> String {
        let mut result = String::with_capacity(5*self.diagram.momentum_labels.len());
        let mut first: bool = true;
        for (i, coefficient) in self.propagator.momentum.iter().enumerate() {
            if *coefficient == 0 { continue; }
            match *coefficient {
                1 => {
                    if !first {
                        write!(&mut result, "+").unwrap();
                    } else {
                        first = false;
                    }
                    write!(&mut result, "{}", self.diagram.momentum_labels[i]).unwrap();
                },
                -1 => {
                    write!(&mut result, "-{}", self.diagram.momentum_labels[i]).unwrap();
                },
                x if x < 0 => write!(&mut result, "-{}*{}", x.abs(), self.diagram.momentum_labels[i]).unwrap(),
                x => write!(&mut result, "{}*{}", x, self.diagram.momentum_labels[i]).unwrap()
            }
        }
        return result;
    }
}

impl std::fmt::Display for PropagatorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[{} -> {}], p = {},",
                 self.particle().get_name(),
                 self.propagator.vertices[0],
                 self.propagator.vertices[1],
                 self.momentum_str()
        )?;
        Ok(())
    }
}

pub struct VertexView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) diagram: &'a DiagramView<'a>,
    pub(crate) vertex: &'a Vertex,
    pub(crate) index: usize,
}

impl<'a> VertexView<'a> {

    /// Get an iterator over the propagators connected to the vertex
    pub fn propagators(&self) -> impl Iterator<Item = Either<LegView<'a>, PropagatorView<'a>>> {
        return self.vertex.propagators.iter().map(
            |i| if *i >= 0 {
                Either::Right(self.diagram.propagator(*i as usize))
            } else {
                let index = (*i + self.diagram.n_ext() as isize) as usize;
                let leg = if index < self.diagram.diagram.incoming_legs.len() {
                    &self.diagram.diagram.incoming_legs[index]
                } else {
                    &self.diagram.diagram.outgoing_legs[index - self.diagram.diagram.incoming_legs.len()]
                };
                Either::Left(LegView {
                    model: self.model,
                    diagram: self.diagram,
                    leg,
                    leg_index: index
                })
            }
        );
    }

    /// Get an iterator over the propagators connected to the vertex ordered like the particles in the interaction
    pub fn propagators_ordered(&self) -> impl Iterator<Item = Either<LegView<'a>, PropagatorView<'a>>> {
        let views = self.propagators().collect_vec();
        let mut perm = Vec::with_capacity(self.vertex.propagators.len());
        let mut seen = vec![false; self.vertex.propagators.len()];
        for ref_particle in self.model.vertex(self.vertex.interaction).particles.iter() {
            for (i, part) in views.iter().map(
                |view| either::for_both!(view, p => p.particle())
            ).enumerate() {
                if !seen[i] && part.get_name() == ref_particle {
                    perm.push(i);
                    seen[i] = true;
                } else {
                    continue;
                }
            }
        }
        return perm.into_iter().map(move |i| views[i].clone());
    }

    /// Get the interaction assigned to the vertex
    pub fn interaction(&self) -> &InteractionVertex {
        return self.model.vertex(self.vertex.interaction);
    }

    /// Check whether the given particle names match the interaction of the vertex
    pub fn match_particles<'q>(&self, query: impl IntoIterator<Item = &'q String>) -> bool {
        return self.model.vertex(self.vertex.interaction).particles.iter().sorted()
            .zip(query.into_iter().sorted()).all(
            |(part, query)| *part == *query
        );
    }
}

impl std::fmt::Display for VertexView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[ {}: ", self.index, self.model.vertex(self.vertex.interaction).name)?;
        for p in self.model.vertex(self.vertex.interaction).particles.iter() {
            write!(f, "{} ", p)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}