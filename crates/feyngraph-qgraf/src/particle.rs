use model::{ParticleBase, ParticleDraw};
use util::HashMap;

use crate::ConstValue;

#[derive(Debug, Clone, PartialEq)]
pub struct QGRAFParticle {
    pub name: String,
    pub anti_name: String,
    pub id: i64,
    pub fermi: bool,
    pub self_anti: bool,
    pub optional_keywords: Vec<String>,
    pub constants: HashMap<String, ConstValue>,
}

impl QGRAFParticle {
    pub(crate) fn new(
        name: impl Into<String>,
        anti_name: impl Into<String>,
        id: i64,
        fermi: bool,
        optional_keywords: Vec<String>,
        constants: HashMap<String, ConstValue>,
    ) -> Self {
        let name = name.into();
        let anti_name = anti_name.into();
        let self_anti = name == anti_name;
        Self {
            name: name,
            anti_name: anti_name,
            id,
            fermi,
            self_anti,
            optional_keywords,
            constants,
        }
    }

    pub fn into_anti(&self) -> Self {
        Self {
            name: self.anti_name.clone(),
            anti_name: self.name.clone(),
            id: if self.self_anti { self.id } else { -self.id },
            fermi: self.fermi,
            self_anti: self.self_anti,
            optional_keywords: self.optional_keywords.clone(),
            constants: self.constants.clone(),
        }
    }
}

impl ParticleBase for QGRAFParticle {
    fn name(&self) -> &str {
        &self.name
    }

    fn anti_name(&self) -> &str {
        &self.anti_name
    }

    fn id(&self) -> i64 {
        self.id
    }

    fn is_self_anti(&self) -> bool {
        self.self_anti
    }

    fn is_fermi(&self) -> bool {
        self.fermi
    }
}

impl ParticleDraw for QGRAFParticle {
    fn display_name(&self) -> &str {
        &self.name
    }

    fn linestyle(&self) -> model::LineStyle {
        model::LineStyle::None
    }
}
