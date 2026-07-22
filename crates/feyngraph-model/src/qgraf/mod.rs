use crate::{ModelBase, ModelError, ParticleBase, VertexBase};
use itertools::Itertools;
use log;
use peg;
use std::path::Path;
use util::{HashMap, IndexMap};

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Int(isize),
    String(String),
    List(Vec<ConstValue>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct QGRAFParticle {
    name: String,
    anti_name: String,
    id: isize,
    fermi: bool,
    self_anti: bool,
    optional_keywords: Vec<String>,
    constants: HashMap<String, ConstValue>,
}

impl QGRAFParticle {
    fn new(
        name: impl Into<String>,
        anti_name: impl Into<String>,
        id: isize,
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

    fn into_anti(&self) -> Self {
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

    pub fn optional_keywords(&self) -> &[String] {
        &self.optional_keywords
    }

    pub fn consts(&self) -> &HashMap<String, ConstValue> {
        &self.constants
    }
}

impl ParticleBase for QGRAFParticle {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> isize {
        todo!()
    }

    fn is_self_anti(&self) -> bool {
        self.self_anti
    }

    fn is_fermi(&self) -> bool {
        self.fermi
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QGRAFVertex {
    name: String,
    particles: Vec<String>,
    couplings: HashMap<String, usize>,
    constants: HashMap<String, ConstValue>,
    spin_map: Vec<isize>,
}

impl QGRAFVertex {
    fn new(
        name: impl Into<String>,
        particles: Vec<String>,
        couplings: HashMap<String, usize>,
        constants: HashMap<String, ConstValue>,
        spin_map: Vec<isize>,
    ) -> Self {
        Self {
            name: name.into(),
            particles,
            couplings,
            constants,
            spin_map,
        }
    }

    pub fn consts(&self) -> &HashMap<String, ConstValue> {
        &self.constants
    }
}

impl VertexBase for QGRAFVertex {
    fn name(&self) -> &str {
        &self.name
    }

    fn particles(&self) -> &[impl AsRef<str>] {
        &self.particles
    }

    fn coupling_orders(&self) -> &HashMap<String, usize> {
        &self.couplings
    }

    fn fermi_map(&self, in_ray: usize) -> usize {
        self.spin_map[in_ray].try_into().unwrap()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct QGRAFModel {
    particles: IndexMap<String, QGRAFParticle>,
    vertices: IndexMap<String, QGRAFVertex>,
    couplings: Vec<String>,
    anti_map: Vec<usize>,
    supports_sign: bool,
}

impl QGRAFModel {
    pub fn parse(path: &Path) -> Result<Self, ModelError> {
        parse_qgraf_model(path)
    }

    pub(crate) fn new(
        particles: IndexMap<String, QGRAFParticle>,
        vertices: IndexMap<String, QGRAFVertex>,
        couplings: Vec<String>,
    ) -> Self {
        let anti_map = particles
            .values()
            .enumerate()
            .map(|(i, p)| {
                if p.self_anti {
                    i
                } else {
                    particles
                        .values()
                        .find_position(|q| q.name == p.anti_name)
                        .as_ref()
                        .unwrap()
                        .0
                }
            })
            .collect_vec();
        return Self {
            particles,
            vertices,
            couplings,
            anti_map,
            supports_sign: true,
        };
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

    fn anti_particle(&self, particle: usize) -> &Self::Particle {
        &self.particles[self.anti_map[particle]]
    }

    fn anti_particle_index(&self, particle: usize) -> usize {
        self.anti_map[particle]
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

#[derive(Debug)]
enum Value<'a> {
    Int(isize),
    String(&'a str),
    List(Vec<Value<'a>>),
}

impl<'a> Value<'a> {
    fn to_owned(self) -> Option<ConstValue> {
        match self {
            Self::Int(i) => Some(ConstValue::Int(i)),
            Self::String(s) => Some(ConstValue::String(s.to_owned())),
            Self::List(v) => Some(ConstValue::List(
                v.into_iter().map(|x| x.to_owned()).flatten().collect(),
            )),
        }
    }
}

enum ModelEntry<'a> {
    Prop(QGRAFParticle),
    Vert(QGRAFVertex),
    Misc(&'a str),
}

peg::parser!(
    grammar qgraf_model() for str {
        rule whitespace() = quiet!{[' ' | '\t' | '\n']}
        rule comment() = quiet!{"%" [^'\n']* "\n"}

        rule _() = quiet!{(comment() / whitespace())*}

        rule alphanumeric() = quiet!{['a'..='z' | 'A'..='Z' | '0'..='9' | '_']}

        rule fermi() -> bool = sign:$(['+' | '-'] / "+1" / "-1") {
            match sign {
                "+" | "+1" => false,
                "-" | "-1" => true,
                _ => unreachable!()
            }
        }
        rule name() -> &'input str = $(alphanumeric()+)
        rule int() -> Value<'input> = int:$(['+' | '-']? ['0'..='9']+) {?
            match int.parse() {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => Err("int")
            }
        }
        rule string() -> Value<'input> = s:$(("\"" [^ '"' ]* "\"") / ("\'" [^ '\'' ]* "\'")) {
            Value::String(&s[1..s.len()-1])
        }

        rule keywords() -> Vec<&'input str> = name() ** (_ "," _)

        rule value() -> Value<'input> = int() / string()
        rule property_value() -> Value<'input> =
            value()
            / "(" _ vals:(value() ** (_ "," _)) _ ")" { Value::List(vals) }

        rule property() -> (&'input str, Value<'input>) = prop:name() _ "=" _ value:property_value() {(prop, value)}

        rule propagator(particle_counter: &mut isize) -> QGRAFParticle =
            "[" _ name:name() _ "," _ anti_name:name() _ "," _ fermi:fermi() _ keywords:("," _ kw:keywords() {kw})? _
            props:(";" _ props:(property()** (_ "," _))? {props} )? _ "]" {?
                let keywords = keywords.unwrap_or(Vec::new()).into_iter().map(|s| s.to_owned()).collect();
                let constants = if let Some(Some(props)) = props {
                    props.into_iter().map(|(k, v)| v.to_owned().map(
                        |v| {
                            if let ConstValue::String(ref s) = v && let Ok(n) = s.parse::<isize>() {
                                (k.into(), ConstValue::Int(n))
                            } else {
                                (k.into(), v)
                            }
                        }
                    )).flatten().collect()
                } else {
                    HashMap::default()
                };
                let id = *particle_counter;
                *particle_counter += 1;
                Ok(QGRAFParticle::new(
                    name,
                    anti_name,
                    id,
                    fermi,
                    keywords,
                    constants
                ))
            }

        rule vertex(vertex_counter: &mut usize) -> QGRAFVertex =
            pos: position!() "[" _ fields:(name() **<3,> (_ "," _)) _
            couplings:(";" _ couplings:(property() ** (_ "," _))? {couplings})? _ "]" {?
                let mut coupling_map = HashMap::default();
                let mut constants = HashMap::default();
                let mut vertex_name = format!("V_{}", vertex_counter);
                if let Some(Some(couplings)) = couplings {
                    for (coupling, value) in couplings {
                        match value {
                            Value::Int(n) => {
                                if n < 0 {
                                    log::warn!("Vertex at position '{}' has negative order in coupling '{}'\
                                        , ignoring this coupling", pos, coupling);
                                    continue;
                                }
                                if let Some(v) =  coupling_map.insert(String::from(coupling), n.try_into().or(Err("Non-negative int"))?) {
                                    log::warn!("Coupling '{}' appears more than once, overwriting previous value", coupling);
                                }
                            },
                            Value::String(s) => {
                                if let Ok(n) = s.parse::<isize>() {
                                    if n < 0 {
                                        log::warn!("Vertex at position '{}' has negative order in coupling '{}'\
                                            , ignoring this coupling", pos, coupling);
                                        continue;
                                    }
                                    if let Some(v) = coupling_map.insert(String::from(coupling), n.try_into().or(Err("Non-negative int"))?) {
                                        log::warn!("Coupling '{}' appears more than once, overwriting previous value", coupling);
                                    }
                                } else {
                                    constants.insert(String::from(coupling), ConstValue::String(s.to_owned()));
                                }
                            }
                            l @ Value::List(_) => { if let Some(l) = l.to_owned() { constants.insert(coupling.into(), l); } }
                            _ => (),
                        }
                    }
                }
                *vertex_counter += 1;
                Ok(QGRAFVertex::new(
                    vertex_name,
                    fields.iter().map(|f| String::from(*f)).collect_vec(),
                    coupling_map,
                    constants,
                    Vec::new()
                ))
            }

        rule misc() -> &'input str =
            s:$("[" [^ ']']* "]")

        pub rule qgraf_model(particle_counter: &mut isize, vertex_counter: &mut usize) -> QGRAFModel =
            _ entries:(
                (
                    p:propagator(particle_counter) {ModelEntry::Prop(p)}
                    / v:vertex(vertex_counter) {ModelEntry::Vert(v)}
                    / m:misc() {ModelEntry::Misc(m)}
                ) ** _
            ) _ {
                let mut particles = IndexMap::default();
                let mut vertices = IndexMap::default();
                let mut couplings = Vec::new();

                for entry in entries {
                    match entry {
                        ModelEntry::Prop(p) => {
                            particles.insert(p.name.clone(), p.clone());
                            if !p.self_anti {
                                let anti_name = p.anti_name.clone();
                                particles.insert(anti_name, p.into_anti());
                            }
                        },
                        ModelEntry::Vert(v) => {
                            for coupling in v.couplings.keys() {
                                if !couplings.contains(coupling) {
                                    couplings.push(coupling.clone());
                                }
                            }
                            vertices.insert(v.name.clone(), v);
                        },
                        ModelEntry::Misc(m) => {
                            log::warn!("Ignoring misc model statement: '{}'", m);
                        }
                    }
                }

                QGRAFModel::new(
                    particles,
                    vertices,
                    couplings,
                )
            }
    }
);

pub(crate) fn parse_qgraf_model(path: &Path) -> Result<QGRAFModel, ModelError> {
    let content = match std::fs::read_to_string(path) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::io(path.display(), e));
        }
    };
    let mut particle_counter: isize = 1;
    let mut vertex_counter: usize = 1;
    return match qgraf_model::qgraf_model(&content, &mut particle_counter, &mut vertex_counter) {
        Ok(mut m) => {
            build_spin_maps(&mut m);
            Ok(m)
        }
        Err(e) => Err(ModelError::parse_error(path.display(), Box::new(e))),
    };
}

fn build_spin_maps(model: &mut QGRAFModel) {
    for v in model.vertices.values_mut() {
        let mut fermions = v
            .particles
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if model.particles.get(s).unwrap().fermi {
                    Some(i)
                } else {
                    None
                }
            })
            .collect_vec();
        if !fermions.is_empty() {
            if fermions.len() > 2 {
                log::warn!(
                    "Ambiguous spin flow mapping for vertex '{}', diagram signs are disabled for this model!",
                    v.name
                );
                model.supports_sign = false;
            }
            let mut spin_map = vec![-1; v.particles.len()];
            while let Some(index) = fermions.pop() {
                let out_leg = fermions.pop().unwrap();
                spin_map[index] = out_leg as isize;
                spin_map[out_leg] = index as isize;
            }
            v.spin_map.extend(spin_map.into_iter().map(|i| i));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::{collections::HashMap, path::PathBuf};
    use test_log::test;

    #[test]
    fn qgraf_qcd_test() {
        let path = PathBuf::from("../../tests/models/qcd.qgraf");
        let model = parse_qgraf_model(&path).unwrap();
        let model_ref = QGRAFModel::new(
            IndexMap::from_iter([
                (
                    String::from("quark"),
                    QGRAFParticle::new(
                        "quark",
                        "antiquark",
                        1,
                        true,
                        Vec::new(),
                        HashMap::from_iter([
                            ("TWOSPIN".into(), ConstValue::Int(1)),
                            ("COLOR".into(), ConstValue::Int(3)),
                        ]),
                    ),
                ),
                (
                    String::from("antiquark"),
                    QGRAFParticle::new(
                        "antiquark",
                        "quark",
                        -1,
                        true,
                        Vec::new(),
                        HashMap::from_iter([
                            ("TWOSPIN".into(), ConstValue::Int(1)),
                            ("COLOR".into(), ConstValue::Int(3)),
                        ]),
                    ),
                ),
                (
                    String::from("gluon"),
                    QGRAFParticle::new(
                        "gluon",
                        "gluon",
                        2,
                        false,
                        Vec::new(),
                        HashMap::from_iter([
                            ("TWOSPIN".into(), ConstValue::Int(2)),
                            ("COLOR".into(), ConstValue::Int(8)),
                        ]),
                    ),
                ),
                (
                    String::from("ghost"),
                    QGRAFParticle::new(
                        "ghost",
                        "antighost",
                        3,
                        true,
                        Vec::new(),
                        HashMap::from_iter([
                            ("TWOSPIN".into(), ConstValue::Int(-2)),
                            ("COLOR".into(), ConstValue::Int(8)),
                        ]),
                    ),
                ),
                (
                    String::from("antighost"),
                    QGRAFParticle::new(
                        "antighost",
                        "ghost",
                        -3,
                        true,
                        Vec::new(),
                        HashMap::from_iter([
                            ("TWOSPIN".into(), ConstValue::Int(-2)),
                            ("COLOR".into(), ConstValue::Int(8)),
                        ]),
                    ),
                ),
            ]),
            IndexMap::from_iter([
                (
                    "V_1".to_string(),
                    QGRAFVertex::new(
                        "V_1".to_string(),
                        vec!["antiquark".to_string(), "quark".to_string(), "gluon".to_string()],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                        HashMap::from_iter([(
                            "CONJ".into(),
                            ConstValue::List(vec![ConstValue::String("+".into()), ConstValue::String("-".into())]),
                        )]),
                        vec![1, 0, -1],
                    ),
                ),
                (
                    "V_2".to_string(),
                    QGRAFVertex::new(
                        "V_2".to_string(),
                        vec!["gluon".to_string(); 3],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                        HashMap::default(),
                        vec![],
                    ),
                ),
                (
                    "V_3".to_string(),
                    QGRAFVertex::new(
                        "V_3".to_string(),
                        vec!["gluon".to_string(); 4],
                        HashMap::from_iter([("QCD".to_string(), 2)]),
                        HashMap::default(),
                        vec![],
                    ),
                ),
                (
                    "V_4".to_string(),
                    QGRAFVertex::new(
                        "V_4".to_string(),
                        vec!["antighost".to_string(), "ghost".to_string(), "gluon".to_string()],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                        HashMap::default(),
                        vec![1, 0, -1],
                    ),
                ),
            ]),
            vec!["QCD".to_string()],
        );
        assert_eq!(model, model_ref);
    }

    #[test]
    fn qgraf_sm_test() {
        let model = parse_qgraf_model(Path::new("../../tests/models/sm.qgraf"));
        match &model {
            Ok(m) => assert!(
                m.particle_by_name("g")
                    .unwrap()
                    .optional_keywords()
                    .contains(&"notadpole".into())
            ),
            Err(e) => {
                println!("{:#?}", e);
            }
        }
        assert!(model.is_ok());
    }
}
