use std::collections::HashMap;
use itertools::Itertools;

mod ufoparser;

#[derive(PartialEq, Debug, Hash)]
pub enum LineStyle {
    Dashed,
    Dotted,
    Straight,
    Wavy,
    Curly,
    Scurly,
    Swavy,
    Double
}

#[derive(Debug, PartialEq, Hash)]
pub struct Particle {
    name: String,
    pdg_code: isize,
    texname: String,
    antitexname: String,
    linestyle: LineStyle,
}

impl Particle {
    pub fn new(name: impl Into<String>,
               pdg_code: isize,
               texname: impl Into<String>,
               antitexname: impl Into<String>,
               linestyle: LineStyle
    ) -> Self {
        return Self {
            name: name.into(),
            pdg_code,
            texname: texname.into(),
            antitexname: antitexname.into(),
            linestyle,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Coupling {
    name: String,
    coupling_orders: HashMap<String, isize>
}

impl Coupling {
    pub fn new(name: impl Into<String>, coupling_orders: HashMap<String, isize>) -> Self {
        return Self {
            name: name.into(),
            coupling_orders,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Vertex {
    name: String,
    particles: Vec<Particle>,
    couplings: Vec<Coupling>,
}

#[derive(Debug, PartialEq)]
pub struct Model {
    particles: HashMap<String, Particle>,
    vertices: Vec<Vertex>,
}

pub struct TopologyModel {
    vertex_degrees: Vec<usize>
}

impl From<&Model> for TopologyModel {
    fn from(model: &Model) -> Self {
        let mut vertex_degrees = Vec::new();
        for vertex in model.vertices.iter() {
            vertex_degrees.push(vertex.particles.len());
        }
        return Self {
            vertex_degrees: vertex_degrees.into_iter().dedup().sorted().collect_vec(),
        }
    }
}

impl From<Vec<usize>> for TopologyModel {
    fn from(vec: Vec<usize>) -> Self {
        return Self {
            vertex_degrees: vec
        }
    }
}

impl TopologyModel {
    pub fn get(&self, i: usize) -> usize {
        return self.vertex_degrees[i];
    }
    
    pub fn degrees_iter(&self) -> impl Iterator<Item=usize> {
        return self.vertex_degrees.clone().into_iter();
    }
}