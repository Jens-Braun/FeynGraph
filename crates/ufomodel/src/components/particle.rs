use model::{LineStyle, ParticleBase, ParticleDraw};

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Particle {
    pub pdg_code: i64,
    pub name: String,
    pub anti_name: String,
    pub spin: i64,
    pub color: i64,
    pub mass: String,
    pub width: String,
    pub texname: String,
    pub anti_texname: String,
    pub charge: f64,
    pub line: LineStyle,
    pub propagating: bool,
    pub goldstone: bool,
    // counterterm: Option<?> // The `counterterm` field is mentioned in the UFO2 paper, but never properly defined or used
    pub propagator: Option<String>,

    pub self_anti: bool,
    pub fermi: bool,
}

impl ParticleBase for Particle {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> i64 {
        self.pdg_code
    }

    fn is_self_anti(&self) -> bool {
        self.self_anti
    }

    fn is_fermi(&self) -> bool {
        self.fermi
    }
}

impl ParticleDraw for Particle {
    fn display_name(&self) -> &str {
        &self.texname
    }

    fn linestyle(&self) -> model::LineStyle {
        todo!()
    }
}
