use std::path::Path;

use log::warn;
use model::{ModelBase, ModelError, ParticleBase};
use util::{HashMap, IndexMap, parsing::diagnostic::DiagnosticContainer};

pub use particle::QGRAFParticle;
pub use vertex::QGRAFVertex;

mod diagnostic;
mod lexer;
mod parser;
mod particle;
pub(crate) mod vertex;

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Int(i64),
    String(String),
    List(Vec<ConstValue>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct QGRAFModel {
    particles: IndexMap<String, QGRAFParticle>,
    vertices: IndexMap<String, QGRAFVertex>,
    couplings: Vec<String>,
    supports_sign: bool,
}

impl QGRAFModel {
    pub fn read(path: &Path) -> Result<DiagnosticContainer<Self>, std::io::Error> {
        let source = std::fs::read_to_string(path)?;
        let name = path
            .file_name()
            .map(|s| s.to_str().unwrap_or("QGRAF Model"))
            .unwrap_or("QGRAF Model");
        Ok(parser::QGRAFParser::parse(name, &source))
    }

    pub fn read_str(name: &str, source: &str) -> DiagnosticContainer<Self> {
        parser::QGRAFParser::parse(name, source)
    }

    pub fn new() -> Self {
        Self {
            particles: IndexMap::default(),
            vertices: IndexMap::default(),
            couplings: Vec::new(),
            supports_sign: true,
        }
    }

    pub fn n_particles(&self) -> usize {
        self.particles.len()
    }

    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn push_particle(
        &mut self,
        name: impl Into<String>,
        anti_name: impl Into<String>,
        fermi: bool,
        optional_keywords: Vec<String>,
        constants: HashMap<String, ConstValue>,
    ) {
        let id = self.particles.len();
        let name = name.into();
        let anti_name = anti_name.into();
        let p = QGRAFParticle::new(
            name.clone(),
            anti_name.clone(),
            id as i64,
            fermi,
            optional_keywords,
            constants,
        );
        let mut orphaned = None;
        if p.self_anti {
            match self.particles.insert(name.clone().into(), p) {
                None => (),
                Some(p_old) => {
                    warn!("Particle `{}` already present in model, replacing.", &name);
                    if !p_old.self_anti {
                        let p_index = self.particles.get_index_of(&p_old.name).unwrap();
                        orphaned = Some(vec![
                            self.particle_index_by_name(self.particle(p_index).anti_name()).unwrap(),
                        ]);
                    }
                }
            }
        } else {
            match self.particles.insert(name.clone().into(), p.clone()) {
                None => match self.particles.insert(anti_name.clone().into(), p.into_anti()) {
                    None => (),
                    Some(anti_old) => {
                        warn!("Particle `{}` already present in model, replacing.", &anti_name);
                        let anti_index = self.particles.get_index_of(&anti_old.name).unwrap();
                        if !anti_old.self_anti {
                            orphaned = Some(vec![
                                self.particle_index_by_name(self.particle(anti_index).anti_name())
                                    .unwrap(),
                            ]);
                        }
                    }
                },
                Some(p_old) => {
                    warn!("Particle `{}` already present in model, replacing.", &name);
                    let p_index = self.particles.get_index_of(&p_old.name).unwrap();
                    match self.particles.insert(anti_name.clone().into(), p.into_anti()) {
                        None => {
                            if !p_old.self_anti {
                                orphaned = Some(vec![
                                    self.particle_index_by_name(self.particle(p_index).anti_name()).unwrap(),
                                ]);
                            }
                        }
                        Some(anti_old) => {
                            warn!("Particle `{}` already present in model, replacing.", &anti_name);
                            let anti_index = self.particles.get_index_of(&anti_old.name).unwrap();
                            let p_anti = self.particle_index_by_name(self.particle(p_index).anti_name()).unwrap();
                            if p_anti != anti_index {
                                orphaned = Some(vec![
                                    p_anti,
                                    self.particle_index_by_name(self.particle(anti_index).anti_name())
                                        .unwrap(),
                                ]);
                            }
                        }
                    }
                }
            }
        }
        if let Some(mut orphans) = orphaned {
            orphans.sort();
            for orphan in orphans.iter().rev() {
                warn!(
                    "Particle `{}` orphaned due to replacement, removing.",
                    self.particles[*orphan].name
                );
                self.particles.shift_remove_index(*orphan);
            }
        }
    }

    pub fn push_vertex(
        &mut self,
        name: impl Into<String>,
        particles: Vec<String>,
        couplings: HashMap<String, usize>,
        constants: HashMap<String, ConstValue>,
    ) -> Result<(), ModelError> {
        let name = name.into();
        for c in couplings.keys() {
            if !self.couplings.contains(c) {
                self.couplings.push(c.clone());
            }
        }
        let fermi_indices = particles
            .iter()
            .map(|p| self.particle_by_name(p).map(|p| p.fermi))
            .collect::<Result<Vec<_>, _>>()?;
        let fermi_indices = fermi_indices
            .into_iter()
            .enumerate()
            .filter_map(|(i, b)| if b { Some(i) } else { None })
            .collect::<Vec<_>>();
        if fermi_indices.len() > 2 {
            warn!(
                "Vertex `{name}` has more than two anti-commuting fields attached, disabling diagram signs for this model."
            );
            self.supports_sign = false;
        }
        let mut fermi_map = vec![-1; particles.len()];
        if fermi_indices.len() == 1 {
            warn!(
                "Vertex `{name}` has a single anti-commuting field attached, disabling diagram signs for this model."
            );
            self.supports_sign = false;
        } else if fermi_indices.len() == 2 {
            fermi_map[fermi_indices[0]] = fermi_indices[1] as i64;
            fermi_map[fermi_indices[1]] = fermi_indices[0] as i64;
        }
        let v = QGRAFVertex::new(name.clone(), particles, couplings, constants, fermi_map);
        match self.vertices.insert(name.clone(), v) {
            None => (),
            Some(_) => log::warn!("Vertex {name} already present in model, replacing."),
        }
        Ok(())
    }
}

impl ModelBase for QGRAFModel {
    type Particle = QGRAFParticle;
    type Vertex = QGRAFVertex;

    fn particle(&self, index: usize) -> &Self::Particle {
        &self.particles[index]
    }

    fn particle_by_name(&self, name: impl AsRef<str>) -> Result<&Self::Particle, ModelError> {
        return self
            .particles
            .get(name.as_ref())
            .ok_or_else(|| ModelError::particle_not_found(name.as_ref()));
    }

    fn particle_index_by_name(&self, name: impl AsRef<str>) -> Result<usize, ModelError> {
        return self
            .particles
            .get_index_of(name.as_ref())
            .ok_or_else(|| ModelError::particle_not_found(name.as_ref()));
    }

    fn vertex(&self, index: usize) -> &Self::Vertex {
        &self.vertices[index]
    }

    fn particles(&self) -> impl ExactSizeIterator<Item = &Self::Particle> {
        self.particles.values()
    }

    fn vertices(&self) -> impl ExactSizeIterator<Item = &Self::Vertex> {
        self.vertices.values()
    }

    fn couplings(&self) -> impl ExactSizeIterator<Item = impl AsRef<str>> {
        self.couplings.iter()
    }

    fn supports_sign(&self) -> bool {
        self.supports_sign
    }
}
