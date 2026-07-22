use crate::{LineStyle, ModelBase, ModelError, ParticleBase, ParticleDraw, VertexBase};
use itertools::Itertools;
use log;
use peg;
use peg::Parse;
use std::path::Path;
use util::{HashMap, IndexMap, contract_indices};

const SM_PARTICLES: &str = include_str!("../Standard_Model/particles.py");
const SM_COUPLING_ORDERS: &str = include_str!("../Standard_Model/coupling_orders.py");
const SM_COUPLINGS: &str = include_str!("../Standard_Model/couplings.py");
const SM_LORENTZ: &str = include_str!("../Standard_Model/lorentz.py");
const SM_VERTICES: &str = include_str!("../Standard_Model/vertices.py");

#[derive(Debug, PartialEq, Hash, Clone, Eq)]
pub struct UFOParticle {
    pub name: String,
    pub anti_name: String,
    pub spin: isize,
    pub color: isize,
    pub pdg_code: isize,
    pub texname: String,
    pub antitexname: String,
    pub linestyle: LineStyle,
    pub self_anti: bool,
    pub fermi: bool,
}

impl UFOParticle {
    pub(crate) fn new(
        name: impl Into<String>,
        anti_name: impl Into<String>,
        spin: isize,
        color: isize,
        pdg_code: isize,
        texname: impl Into<String>,
        antitexname: impl Into<String>,
        linestyle: LineStyle,
        fermi: bool,
    ) -> Self {
        let texname = texname.into();
        let antitexname = antitexname.into();
        let self_anti = texname == antitexname;
        return Self {
            name: name.into(),
            anti_name: anti_name.into(),
            spin,
            color,
            pdg_code,
            texname,
            antitexname,
            linestyle,
            self_anti,
            fermi,
        };
    }
    pub(crate) fn into_anti(self) -> Self {
        return Self {
            name: self.anti_name,
            anti_name: self.name,
            spin: -self.spin,
            color: -self.color,
            pdg_code: -self.pdg_code,
            texname: self.antitexname,
            antitexname: self.texname,
            linestyle: self.linestyle,
            self_anti: self.self_anti,
            fermi: self.fermi,
        };
    }
}

impl ParticleBase for UFOParticle {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> isize {
        self.pdg_code
    }

    fn is_self_anti(&self) -> bool {
        self.self_anti
    }

    fn is_fermi(&self) -> bool {
        self.fermi
    }
}

impl ParticleDraw for UFOParticle {
    fn display_name(&self) -> &str {
        &self.texname
    }

    fn linestyle(&self) -> LineStyle {
        self.linestyle
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct UFOVertex {
    pub name: String,
    pub particles: Vec<String>,
    pub spin_map: Vec<isize>,
    pub coupling_orders: HashMap<String, usize>,
}

impl UFOVertex {
    pub(crate) fn new(
        name: String,
        particles: Vec<String>,
        spin_map: Vec<isize>,
        coupling_orders: HashMap<String, usize>,
    ) -> Self {
        return Self {
            name,
            particles,
            spin_map,
            coupling_orders,
        };
    }

    pub fn add_coupling(&mut self, coupling: impl Into<String> + Clone, power: usize) {
        match self.coupling_orders.insert(coupling.clone().into(), power) {
            None => (),
            Some(c) => log::warn!(
                "Vertex already has power {} in coupling {}, overwriting.",
                c,
                coupling.into()
            ),
        }
    }
}

impl VertexBase for UFOVertex {
    fn name(&self) -> &str {
        &self.name
    }

    fn particles(&self) -> &[impl AsRef<str>] {
        &self.particles
    }

    fn coupling_orders(&self) -> &HashMap<String, usize> {
        &self.coupling_orders
    }

    fn fermi_map(&self, in_ray: usize) -> usize {
        self.spin_map[in_ray] as usize
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct UFOModel {
    particles: IndexMap<String, UFOParticle>,
    vertices: IndexMap<String, UFOVertex>,
    couplings: Vec<String>,
    splittings: HashMap<String, HashMap<String, Vec<(usize, usize)>>>,
    anti_map: Vec<usize>,
}

impl UFOModel {
    pub fn parse(path: &Path) -> Result<Self, ModelError> {
        parse_ufo_model(path)
    }

    pub fn sm() -> Self {
        sm()
    }

    /// Create a new model without any particles, vertices or couplings.
    pub fn empty() -> Self {
        return Self {
            particles: IndexMap::default(),
            vertices: IndexMap::default(),
            couplings: Vec::new(),
            splittings: HashMap::default(),
            anti_map: Vec::new(),
        };
    }

    pub(crate) fn new(
        particles: IndexMap<String, UFOParticle>,
        vertices: IndexMap<String, UFOVertex>,
        couplings: Vec<String>,
        splittings: HashMap<String, HashMap<String, Vec<(usize, usize)>>>,
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
                        .find_position(|q| q.pdg_code == -p.pdg_code)
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
            splittings,
            anti_map,
        };
    }

    /// Add a new particle with the given properties to the model or overwrite an existing one. If `name == anti_name`,
    /// the particle is automatically marked as its own anti particle. Otherwise, the corresponding anti particle is
    /// automatically also added to the model.
    pub fn add_particle<S: Into<String> + PartialEq + Clone>(
        &mut self,
        name: S,
        anti_name: S,
        spin: isize,
        color: isize,
        pdg_code: isize,
        texname: S,
        antitexname: S,
        linestyle: LineStyle,
        fermi: bool,
    ) {
        let p = UFOParticle {
            name: name.clone().into(),
            anti_name: anti_name.clone().into(),
            spin,
            color,
            pdg_code,
            texname: texname.into(),
            antitexname: antitexname.into(),
            self_anti: name == anti_name,
            linestyle,
            fermi,
        };
        if p.self_anti {
            match self.particles.insert(name.clone().into(), p) {
                None => (),
                Some(_) => log::warn!("Particle {} already present in model, replacing.", name.into()),
            }
            self.anti_map.push(self.anti_map.len());
        } else {
            match self.particles.insert(name.clone().into(), p.clone()) {
                None => (),
                Some(_) => log::warn!("Particle {} already present in model, replacing.", name.clone().into()),
            }
            self.particles.insert(anti_name.clone().into(), p.into_anti());
            self.anti_map.push(self.anti_map.len() + 1);
            self.anti_map.push(self.anti_map.len() - 1);
        }
    }

    /// Add a new vertex with the given properties to the model or overwrite an existing one. The `i`-th entry of the
    /// `spin_map` must be the leg `j` to which leg `i` is spin-connected.
    pub fn add_vertex<S: Into<String> + PartialEq + Clone>(
        &mut self,
        name: S,
        particles: Vec<S>,
        spin_map: Vec<isize>,
        coupling_orders: HashMap<S, usize>,
    ) -> Result<(), ModelError> {
        for coupling in coupling_orders.keys() {
            if !self.couplings.contains(&coupling.clone().into()) {
                self.couplings.push(coupling.clone().into());
            }
        }
        let v = UFOVertex::new(
            name.clone().into(),
            particles.into_iter().map(|s| s.into()).collect(),
            spin_map,
            HashMap::from_iter(coupling_orders.into_iter().map(|(k, v)| (k.into(), v))),
        );
        for p in v.particles() {
            if !self.particles.contains_key(p.as_ref()) {
                return Err(ModelError::particle_not_found(p.as_ref()));
            }
        }
        match self.vertices.insert(name.clone().into(), v) {
            None => (),
            Some(_) => log::warn!("Vertex {} already present in model, replacing.", name.into()),
        }
        Ok(())
    }

    /// Deduplicate vertices in the model, i.e. merge all vertices with identical particles, spin connection and
    /// coupling powers. Returns a hash map containing the new vertex and all vertices which were merged into it.
    pub fn merge_vertices(&mut self) -> IndexMap<String, Vec<String>> {
        let mut mergings = IndexMap::default();
        let mut merged_vertices = IndexMap::default();
        let mut i = 1;
        for (_, vertices) in self
            .vertices
            .values()
            .into_group_map_by(|v| {
                (
                    v.particles.clone(),
                    v.coupling_orders.clone().into_iter().collect_vec(),
                    v.spin_map.clone(),
                )
            })
            .into_iter()
            .sorted_by_key(|(x, _)| x.clone())
        {
            if vertices.len() > 1 {
                mergings.insert(format!("V_M_{}", i), vertices.iter().map(|v| v.name.clone()).collect());
                merged_vertices.insert(
                    format!("V_M_{}", i),
                    UFOVertex {
                        name: format!("V_M_{}", i),
                        particles: vertices[0].particles.clone(),
                        spin_map: vertices[0].spin_map.clone(),
                        coupling_orders: vertices[0].coupling_orders.clone(),
                    },
                );
                i += 1;
            } else {
                merged_vertices.insert(vertices[0].name.clone(), vertices[0].clone());
            }
        }
        self.vertices = merged_vertices;
        return mergings;
    }

    /// Add a new coupling to the interaction vertex `vertex` or overwrite an existing one.
    pub fn add_coupling<S: Into<String> + Clone>(
        &mut self,
        vertex: S,
        coupling: S,
        power: usize,
    ) -> Result<(), ModelError> {
        match self.vertices.get_mut(&vertex.clone().into()) {
            Some(v) => v.add_coupling(coupling, power),
            None => {
                return Err(ModelError::vertex_not_found(vertex.into()));
            }
        }
        Ok(())
    }

    /// Split the existing vertex `vertex` into new vertices with names `new_vertices`.
    pub fn split_vertex<S: Into<String> + PartialEq + Clone>(
        &mut self,
        vertex: S,
        new_vertices: &[S],
    ) -> Result<(), ModelError> {
        let v = self.vertices.shift_remove(&vertex.clone().into());
        match v {
            None => {
                return Err(ModelError::vertex_not_found(vertex.into()));
            }
            Some(v) => {
                for name in new_vertices.iter() {
                    let mut new_vertex = v.clone();
                    new_vertex.name = name.clone().into();
                    match self.vertices.insert(name.clone().into(), new_vertex) {
                        None => (),
                        Some(_) => log::warn!("Vertex {} already present in model, replacing.", name.clone().into()),
                    }
                }
            }
        }
        Ok(())
    }
}

impl ModelBase for UFOModel {
    type Particle = UFOParticle;
    type Vertex = UFOVertex;

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
        true
    }
}

#[allow(unused)]
#[derive(Debug)]
enum Value<'a> {
    Int(isize),
    Rational(isize, isize),
    String(&'a str),
    Bool(bool),
    List(Vec<Value<'a>>),
    SIDict(HashMap<String, usize>),
    CODict(HashMap<(usize, usize), String>),
    Particle(UFOParticle),
    None,
}

impl<'a> Value<'a> {
    fn int(self) -> Result<isize, &'static str> {
        match self {
            Self::Int(i) => Ok(i),
            _ => Err("int"),
        }
    }

    fn str(self) -> Result<&'a str, &'static str> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err("string"),
        }
    }

    fn i_list(self) -> Result<Vec<isize>, &'static str> {
        match self {
            Self::List(l) => Ok(l.into_iter().map(|v| v.int()).collect::<Result<Vec<_>, &str>>()?),
            _ => Err("list of ints"),
        }
    }

    fn s_list(self) -> Result<Vec<String>, &'static str> {
        match self {
            Self::List(l) => Ok(l
                .into_iter()
                .map(|v| v.str().map(|s| s.to_owned()))
                .collect::<Result<Vec<_>, &str>>()?),
            _ => Err("list of strings"),
        }
    }

    fn si_dict(self) -> Result<HashMap<String, usize>, &'static str> {
        match self {
            Self::SIDict(d) => {
                return Ok(d);
            }
            _ => Err("string-int dict")?,
        }
    }

    fn co_dict(self) -> Result<HashMap<(usize, usize), String>, &'static str> {
        match self {
            Self::CODict(d) => {
                return Ok(d);
            }
            _ => Err("(int, int, int?)-string dict")?,
        }
    }
}

peg::parser! {
    grammar ufo_model() for str {
        rule whitespace() = quiet!{[' ' | '\t' | '\n' | '\r']}
        rule comment() = quiet!{"#" [^'\n']* "\n"}

        rule _() = quiet!{(comment() / whitespace())*}

        rule alphanumeric() = quiet!{['a'..='z' | 'A'..='Z' | '0'..='9' | '_']}

        rule name() -> &'input str = $(alphanumeric()+)
        rule bool() -> Value<'input> = "True" {Value::Bool(true)} / "False" {Value::Bool(false)}
        rule int() -> Value<'input> = int:$(['+' | '-']? ['0'..='9']+) {?
            match int.parse() {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => Err("int")
            }
        }
        rule rational() -> Value<'input> = num:int() _ "/" _ denom:int() {
            match (num, denom) {
                (Value::Int(num), Value::Int(denom))=> Value::Rational(num, denom),
                _ => unreachable!()
            }
        }
        rule string() -> Value<'input> = "r"? s:$(("\"" [^ '"' ]* "\"") / ("\'" [^ '\'' ]* "\'")) {
            Value::String(&s[1..s.len()-1])
        }
        rule si_dict(input: &str) -> Value<'input> =
            pos: position!() "{" _ entries:((key:string() _ ":" _ val:value(input) {(key, val)}) ** (_ "," _)) _ ","? _ "}" {?
                let mut result = HashMap::default();
                for (k, v) in entries.into_iter().map(
                    |(key, val)| (key.str().unwrap().to_owned(), val.int().map(|i| i.try_into()))
                ) {
                    match v {
                        Ok(Ok(v)) => {
                            if let Some(i) =  result.insert(k.clone(), v) {
                                log::warn!("Value '{}' appears multiple times in dict, keeping only value {}", k, i);
                            }
                        }
                        Ok(Err(_)) => log::warn!("Coupling '{}' has non-negative order at {}, ignoring", k, input.position_repr(pos)),
                        _ => Err("string-int dict")?
                    }
                }
                Ok(Value::SIDict(result))
            }

        rule co_dict() -> Value<'input> =
            "{" _ entries:(("(" _ i:int() _ "," _ j:int() (_ "," _ int())? _ ")" _ ":" _ c:("C." _ c:name() {c}) {((i, j), c)}) ** (_ "," _)) _ ","? _ "}" {?
                let mut result = HashMap::default();
                for ((i, j), c) in entries.into_iter() {
                    let i = i.int()?.try_into().or(Err("non-negative int"))?;
                    let j = j.int()?.try_into().or(Err("non-negative int"))?;
                    if let Some(v) =  result.insert((i, j), c.to_owned()) {
                        log::warn!("Basis element '({}, {})' appears multiple times in dict, keeping only coupling {}", i, j, v);
                    }
                }
                Ok(Value::CODict(result))
            }

        rule p_list() -> Value<'input> =
            "[" _ vals:(("P." _ name:name() {Value::String(name)}) ** (_ "," _)) _ ","? _ "]" {Value::List(vals)}

        rule l_list() -> Value<'input> =
            "[" _ vals:(("L." _ name:name() {Value::String(name)}) ** (_ "," _)) _ ","? _ "]" {Value::List(vals)}

        rule value_list(input: &str) -> Value<'input> =
            "[" _ vals:(property_value(input) ** (_ "," _)) _ ","? _ "]" { Value::List(vals) }

        rule value(input: &str) -> Value<'input> = rational() / int() / bool() / string() / p_list() / l_list() / si_dict(input) / co_dict()
        rule property_value(input: &str) -> Value<'input> =
            value(input)
            / value_list(input)
            / ([^',' | ')' | ']' | '}']* {Value::None})

        rule property(input: &str) -> (&'input str, Value<'input>) = prop:name() _ "=" _ value:property_value(input) {(prop, value)}

        rule particle(input: &str) -> (&'input str, Value<'input>) =
            pos: position!() py_name:name() _ "=" _ "Particle(" _ props:(property(input) **<1,> (_ "," _)) _ ","? _ ")" {?
                let mut pdg_code = None;
                let mut name = None;
                let mut antiname = None;
                let mut texname = None;
                let mut antitexname = None;
                let mut linestyle = None;
                let mut twospin = None;
                let mut color = None;
                for (prop, value) in props {
                    match prop.to_lowercase().as_str() {
                        "name" => name = Some(value.str()?),
                        "antiname" => antiname = Some(value.str()?),
                        "pdg_code" => pdg_code = Some(value.int()?),
                        "spin" => twospin = Some(value.int()? - 1),
                        "color" => color = Some(value.int()?),
                        "texname" => texname = Some(value.str()?),
                        "antitexname" => antitexname = Some(value.str()?),
                        "line" => {
                            linestyle = Some(match value.str()? {
                                "dashed" => LineStyle::Dashed,
                                "dotted" => LineStyle::Dotted,
                                "straight" => LineStyle::Straight,
                                "wavy" => LineStyle::Wavy,
                                "curly" => LineStyle::Curly,
                                "scurly" => LineStyle::Scurly,
                                "swavy" => LineStyle::Swavy,
                                "double" => LineStyle::Double,
                                x => {
                                    log::warn!(
                                        "Option 'linestyle' for particle '{}' at {} has unknown value '{}'",
                                        py_name,
                                        input.position_repr(pos),
                                        x
                                        );
                                    LineStyle::None
                                }
                            })
                        },
                        _ => ()
                    }
                }
                if linestyle.is_none() {
                    linestyle = Some(match (twospin, color) {
                        (Some(0), _) => LineStyle::Dashed,
                        (Some(1), _) if *name.unwrap() != *antiname.unwrap() => LineStyle::Straight,
                        (Some(1), Some(1)) => LineStyle::Swavy,
                        (Some(1), _) => LineStyle::Scurly,
                        (Some(2), Some(1)) => LineStyle::Wavy,
                        (Some(2), _) => LineStyle::Curly,
                        (Some(4), _) => LineStyle::Double,
                        (Some(-2), _) => LineStyle::Dotted,
                        _ => {
                            log::warn!("Unable to determine linestyle for particle '{}' at {} from spin an color", name.unwrap(), input.position_repr(pos));
                            LineStyle::None
                        },
                    });
                }
                let mut fermi = false;
                if let Some(s) = twospin && (s < 0 || s % 2 == 1) {
                    fermi = true;
                }
                Ok((py_name, Value::Particle(UFOParticle::new(
                    name.unwrap(),
                    antiname.unwrap(),
                    twospin.unwrap(),
                    color.unwrap(),
                    pdg_code.unwrap(),
                    texname.unwrap(),
                    antitexname.unwrap(),
                    linestyle.unwrap(),
                    fermi
                ))))
            }

        rule anti_particle() -> (&'input str, Value<'input>) =
            _ anti_ident:name() _ "=" _ p_ident:name() ".anti()" {(anti_ident, Value::String(p_ident))}

        rule coupling_order(input: &str) -> String =
            pos: position!() py_name:name() _ "=" _ "CouplingOrder(" _ props:(property(input) **<1,> (_ "," _)) _ ","? _ ")" {?
                let mut coupling: Option<String> = None;
                for (prop, value) in props {
                    match prop.to_lowercase().as_str() {
                        "name" => coupling = Some(value.str()?.to_owned()),
                        "expansion_order" => (),
                        "hierarchy" => (),
                        _ => {}
                    }
                }
                match coupling {
                    None => Err("coupling order name"),
                    Some(name) => Ok(name)
                }
            }

        rule coupling(input: &str) -> (String, HashMap<String, usize>) =
            pos: position!() py_name:name() _ "=" _ "Coupling(" _ props:(property(input) **<1,> (_ "," _)) _ ","? _ ")" {?
                let mut orders = None;
                for (prop, value) in props {
                    match prop.to_lowercase().as_str() {
                        "value" => (),
                        "name" => (),
                        "order" => orders = Some(value.si_dict()?),
                        x => {
                            log::warn!(
                                "Option '{}' for coupling {} at {} is unknown, ignoring",
                                x,
                                py_name,
                                input.position_repr(pos),
                                );
                        }
                    }
                }
                match orders {
                    None => Err("coupling order dict"),
                    Some(orders) => {
                        return Ok((py_name.to_owned(), orders));
                    }
                }
            }

        rule vertex(
            input: &str,
            couplings: &HashMap<String, HashMap<String, usize>>,
            ident_map: &HashMap<String, String>,
            lorentz_structures: &HashMap<String, Vec<isize>>
        ) -> (Vec<UFOVertex>, (String, HashMap<String, Vec<(usize, usize)>>)) =
            pos: position!() py_name:name() _ "=" _ "Vertex(" _ props:(property(input) **<1,> (_ "," _)) _ ","? _ ")" {?
                return parse_vertex(input, couplings, ident_map, lorentz_structures, py_name, props, pos);
            }

        rule ct_vertex(
            input: &str,
            couplings: &HashMap<String, HashMap<String, usize>>,
            ident_map: &HashMap<String, String>,
            lorentz_structures: &HashMap<String, Vec<isize>>
        ) -> (Vec<UFOVertex>, (String, HashMap<String, Vec<(usize, usize)>>)) =
            pos: position!() py_name:name() _ "=" _ "CTVertex(" _ props:(property(input) **<1,> (_ "," _)) _ ")" {?
                return parse_vertex(input, couplings, ident_map, lorentz_structures, py_name, props, pos);
            }

        rule lorentz_atom() -> (isize, isize) =
            ("Identity" /  "Gamma5" / "ProjM" / "ProjP"  / "C") _
            "(" _ i:int() _ "," _ j:int() _ ")" {? Ok((i.int()? - 1, j.int()? - 1)) }
            / "Gamma" _ "(" _ int() _ "," _ i:int() _ "," _ j:int() _ ")" {? Ok((i.int()? - 1, j.int()? - 1)) }
            / "Sigma" _ "(" _ int() _ "," _ int() _ "," _ i:int() _ "," _ j:int() _ ")" {? Ok((i.int()? - 1, j.int()? - 1)) }

        pub rule lorentz_structure() -> Vec<(isize, isize)> =
            _ connections:((a:lorentz_atom() {Some(a)} / [_] {None})) ** _ {
                connections.into_iter().flatten().collect_vec()
            }

        pub rule lorentz(input: &str) -> (&'input str, Vec<isize>) =
            pos: position!() py_name:name() _ "=" _ "Lorentz(" _ props:(property(input) **<1,> (_ "," _)) _ ","? _ ")" {?
                let mut name = None;
                let mut spins = None;
                let mut structure = None;
                for (prop, value) in props {
                    match prop.to_lowercase().as_str() {
                        "name" => name = Some(value.str()?),
                        "spins" => spins = Some(value.i_list()?),
                        "structure" => structure = Some(value.str()?),
                        x => {
                            log::warn!(
                                "Option '{}' for lorentz structure {} at {} is unknown, ignoring",
                                x,
                                py_name,
                                input.position_repr(pos),
                                );
                        }
                    }
                }
                let mut connections = lorentz_structure(structure.unwrap()).or(Err("lorentz structure"))?;

                let spins = spins.unwrap().into_iter().map(|s| s-1).collect_vec();
                if connections.is_empty() && spins.iter().any(|s| *s < 0 || *s % 2 == 1) {
                    let mut seen = vec![false; spins.len()];
                    for (i, s) in spins.iter().enumerate() {
                        if seen[i] || (*s >= 0 && *s % 2 == 0) {
                            seen[i] = true;
                            continue;
                        }
                        let connected = spins.iter().enumerate()
                            .filter_map(|(j, r)| if *s == *r && i != j {Some(j)} else {None}).collect_vec();
                        if connected.len() > 1 {
                            log::warn!("Ambiguous spin flow mapping for lorentz '{}' at {}, the calculated diagram signs for diagrams \
                                        containing this lorentz structure might be wrong!", name.unwrap(), input.position_repr(pos));
                        }
                        let j = connected[0];
                        seen[i] = true;
                        seen[j] = true;
                        connections.push((i as isize, j as isize));
                    }
                }
                return Ok((py_name, contract_indices(connections)));
            }

        pub rule particles(input: &str) -> Vec<(String, UFOParticle)> =
            _ (!particle(input) [_])* _ particles:((anti_particle() / particle(input)) ** _) _ {?
                let mut res = Vec::with_capacity(particles.len());
                for (py_name, v) in particles.into_iter() {
                    match v {
                        Value::Particle(p) => {res.push((py_name.to_owned(), p))},
                        Value::String(p_ident) => {
                            let p_pos = match res.iter().position(|(ident, _)| ident == p_ident) {
                                Some(i) => i,
                                None => {
                                    return Err("anti-particle of previously declared particle");
                                }
                            };
                            res.insert(p_pos+1, (py_name.to_owned(), res[p_pos].1.clone().into_anti()));
                        },
                        _ => unreachable!()
                    }
                }
                return Ok(res);
            }

        pub rule coupling_orders(input: &str) -> Vec<String> =
        _ (!coupling_order(input) [_])* _ orders:(coupling_order(input) ** _) _ {orders}

        pub rule couplings(input: &str) -> HashMap<String, HashMap<String, usize>> =
        _ (!coupling(input) [_])* _ couplings:(coupling(input) ** _) _ {couplings.into_iter().collect()}

        pub rule lorentz_structures(input: &str) -> HashMap<String, Vec<isize>> =
        _ (!lorentz(input) [_])* _ structures:(lorentz(input) ** _) _ {
            structures.into_iter().map(|(s, v)| (s.to_owned(), v)).collect()
        }

        pub rule vertices(
            input: &str,
            couplings: &HashMap<String, HashMap<String, usize>>,
            ident_map: &HashMap<String, String>,
            lorentz_structures: &HashMap<String, Vec<isize>>
        ) -> (IndexMap<String, UFOVertex>, HashMap<String, HashMap<String, Vec<(usize, usize)>>>) =
        _ (!vertex(input, couplings, ident_map, lorentz_structures) [_])* _
        vertices:(vertex(input, couplings, ident_map, lorentz_structures) ** _) _ {
                let (vertices, splittings): (Vec<Vec<UFOVertex>>, Vec<(String, HashMap<String, Vec<(usize, usize)>>)>) = vertices.into_iter().unzip();
                return (
                    vertices.into_iter().flatten().map(|v| (v.name.clone(), v)).collect(),
                    HashMap::from_iter(splittings.into_iter().filter(|(_, m)| !m.is_empty()))
                )
            }

        pub rule ct_vertices(
            input: &str,
            couplings: &HashMap<String, HashMap<String, usize>>,
            ident_map: &HashMap<String, String>,
            lorentz_structures: &HashMap<String, Vec<isize>>
        ) -> (IndexMap<String, UFOVertex>, HashMap<String, HashMap<String, Vec<(usize, usize)>>>) =
        _ (!ct_vertex(input, couplings, ident_map, lorentz_structures) [_])* _
        vertices:(ct_vertex(input, couplings, ident_map, lorentz_structures) ** _) _ {
                let (vertices, splittings): (Vec<Vec<UFOVertex>>, Vec<(String, HashMap<String, Vec<(usize, usize)>>)>) = vertices.into_iter().unzip();
                return (
                    vertices.into_iter().flatten().filter_map(
                        |v| if v.particles.len() > 2 {Some((v.name.clone(), v))} else {
                            log::warn!("Vertex {} is two-point, which is currently unsupported by FeynGraph. Ignoring.", v.name);
                            None
                        }
                    ).collect(),
                    HashMap::from_iter(splittings.into_iter().filter(|(_, m)| !m.is_empty()))
                )
            }
    }
}

fn parse_vertex(
    input: &str,
    couplings: &HashMap<String, HashMap<String, usize>>,
    ident_map: &HashMap<String, String>,
    lorentz_structures: &HashMap<String, Vec<isize>>,
    py_name: &str,
    props: Vec<(&str, Value<'_>)>,
    pos: usize,
) -> Result<(Vec<UFOVertex>, (String, HashMap<String, Vec<(usize, usize)>>)), &'static str> {
    let mut particles = None;
    let mut name = None;
    let mut coupling_dict = None;
    let mut lorentz_list = None;
    let mut splitting: HashMap<String, Vec<(usize, usize)>> = HashMap::default();

    for (prop, value) in props {
        match prop.to_lowercase().as_str() {
            "name" => name = Some(value.str()?.to_owned()),
            "particles" => particles = Some(value.s_list()?),
            "couplings" => coupling_dict = Some(value.co_dict()?),
            "lorentz" => lorentz_list = Some(value.s_list()?),
            "color" => (),
            "loop_particles" => (),
            "type" => (),
            x => {
                log::warn!(
                    "Option '{}' for vertex '{}' at {} is unknown, ignoring",
                    x,
                    py_name,
                    input.position_repr(pos),
                );
            }
        }
    }
    let particles = particles
        .unwrap()
        .iter()
        .map(|p| ident_map.get(p).unwrap().clone())
        .collect_vec();
    let name = name.unwrap();

    let mut unique_spin_mappings: Vec<(Vec<isize>, Vec<usize>)> = Vec::new();
    for (i, spin_map) in lorentz_list
        .unwrap()
        .iter()
        .map(|s| lorentz_structures.get(s).unwrap())
        .enumerate()
    {
        match unique_spin_mappings.iter_mut().find(|(v, _)| **v == *spin_map) {
            None => unique_spin_mappings.push((spin_map.clone(), vec![i])),
            Some((_, l)) => l.push(i),
        }
    }

    #[allow(clippy::type_complexity)]
    let mut unique_coupling_orders: Vec<(&HashMap<String, usize>, Vec<(usize, usize)>)> = vec![];
    if let Some(coupling_dict) = coupling_dict {
        for ((i, j), c) in coupling_dict.iter() {
            match couplings.get(c) {
                Some(d) => match unique_coupling_orders.iter_mut().find(|(v, _)| **v == *d) {
                    None => unique_coupling_orders.push((d, vec![(*i, *j)])),
                    Some((_, l)) => l.push((*i, *j)),
                },
                None => {
                    log::warn!(
                        "Vertex '{}' at '{}' contains coupling '{}', which is not specified in 'couplings.py', ignoring",
                        py_name,
                        input.position_repr(pos),
                        c
                    );
                }
            }
        }
    } else {
        log::warn!(
            "No couplings specified for vertex {} at {}",
            py_name,
            input.position_repr(pos)
        );
    }

    let mut vertices = Vec::new();
    if unique_coupling_orders.len() > 1 || unique_spin_mappings.len() > 1 {
        for (spin_map, lorentz_indices) in unique_spin_mappings.iter() {
            if unique_coupling_orders.is_empty() {
                splitting.insert(
                    format!("{}_{}", &name, vertices.len()),
                    lorentz_indices.iter().map(|i| (0, *i)).collect(),
                );
                vertices.push(UFOVertex::new(
                    format!("{}_{}", &name, vertices.len()),
                    particles.clone(),
                    spin_map.clone(),
                    HashMap::default(),
                ))
            } else {
                for (d, l) in unique_coupling_orders.iter() {
                    if l.iter().any(|(_, i)| lorentz_indices.contains(i)) {
                        splitting.insert(
                            format!("{}_{}", &name, vertices.len()),
                            l.iter().filter(|(_, i)| lorentz_indices.contains(i)).cloned().collect(),
                        );
                        vertices.push(UFOVertex::new(
                            format!("{}_{}", &name, vertices.len()),
                            particles.clone(),
                            spin_map.clone(),
                            (*d).clone(),
                        ))
                    }
                }
            }
        }
    }

    match (unique_spin_mappings.len(), unique_coupling_orders.len()) {
        (1, 0) => vertices.push(UFOVertex::new(
            name.clone(),
            particles.clone(),
            unique_spin_mappings[0].0.clone(),
            HashMap::default(),
        )),
        (1, 1) => vertices.push(UFOVertex::new(
            name.clone(),
            particles.clone(),
            unique_spin_mappings[0].0.clone(),
            unique_coupling_orders[0].0.clone(),
        )),
        (1, x) => {
            log::warn!(
                "Ambiguous coupling powers for vertex {}, splitting into {} vertices '{}_0' .. '{}_{}'",
                &name,
                x,
                &name,
                &name,
                x - 1
            );
        }
        (x, 1) => {
            log::warn!(
                "Ambiguous spin mapping for vertex {}, splitting into {} vertices '{}_0' .. '{}_{}'",
                &name,
                x,
                &name,
                &name,
                x - 1
            );
        }
        (_, _) => {
            log::warn!(
                "Ambiguous coupling powers and spin mapping for vertex {}, \
                splitting into {} vertices '{}_0' .. '{}_{}'",
                &name,
                vertices.len(),
                &name,
                &name,
                vertices.len() - 1
            );
        }
    }

    Ok((vertices, (name, splitting)))
}

pub fn parse_ufo_model(path: &Path) -> Result<UFOModel, ModelError> {
    let particle_content = match std::fs::read_to_string(path.join("particles.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::io(path.join("particles.py").display(), e));
        }
    };
    let (ident_map, particles): (HashMap<String, String>, IndexMap<String, UFOParticle>) =
        match ufo_model::particles(&particle_content, &particle_content) {
            Ok(x) => x
                .into_iter()
                .map(|(py_name, p)| ((py_name, p.anti_name.clone()), (p.name.clone(), p)))
                .unzip(),
            Err(e) => {
                return Err(ModelError::parse_error("particles.py", Box::new(e)));
            }
        };

    let coupling_order_content = match std::fs::read_to_string(path.join("coupling_orders.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::io(path.join("coupling_orders.py").display(), e));
        }
    };
    let mut coupling_orders = match ufo_model::coupling_orders(&coupling_order_content, &coupling_order_content) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::parse_error("coupling_orders.py", Box::new(e)));
        }
    };

    let coupling_content = match std::fs::read_to_string(path.join("couplings.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::io(path.join("couplings.py").display(), e));
        }
    };
    let mut couplings = match ufo_model::couplings(&coupling_content, &coupling_content) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::parse_error("couplings.py", Box::new(e)));
        }
    };

    let lorentz_content = match std::fs::read_to_string(path.join("lorentz.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::io(path.join("lorentz.py").display(), e));
        }
    };
    let lorentz_structures = match ufo_model::lorentz_structures(&lorentz_content, &lorentz_content) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::parse_error("lorentz.py", Box::new(e)));
        }
    };

    let vertices_content = match std::fs::read_to_string(path.join("vertices.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::io(path.join("vertices.py").display(), e));
        }
    };
    let (mut vertices, mut splittings) = match ufo_model::vertices(
        &vertices_content,
        &vertices_content,
        &couplings,
        &ident_map,
        &lorentz_structures,
    ) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::parse_error("vertices.py", Box::new(e)));
        }
    };

    for v in vertices.values() {
        for coupling in v.coupling_orders.keys() {
            if !coupling_orders.contains(coupling) {
                log::warn!(
                    "Vertex '{}' contains coupling '{}', which is not specified in 'coupling_orders.py'. Adding it now.",
                    &v.name,
                    coupling
                );
                coupling_orders.push(coupling.clone());
            }
        }
    }

    if let Ok(v) = std::fs::exists(path.join("CT_couplings.py"))
        && v == true
    {
        let ct_coupling_content = match std::fs::read_to_string(path.join("CT_couplings.py")) {
            Ok(x) => x,
            Err(e) => {
                return Err(ModelError::io(path.join("CT_couplings.py").display(), e));
            }
        };
        let ct_couplings = match ufo_model::couplings(&ct_coupling_content, &ct_coupling_content) {
            Ok(x) => x,
            Err(e) => {
                return Err(ModelError::parse_error("CT_couplings.py", Box::new(e)));
            }
        };
        couplings.extend(ct_couplings.into_iter());
    }

    if let Ok(v) = std::fs::exists(path.join("CT_vertices.py"))
        && v == true
    {
        let ct_vertices_content = match std::fs::read_to_string(path.join("CT_vertices.py")) {
            Ok(x) => x,
            Err(e) => {
                return Err(ModelError::io(path.join("CT_vertices.py").display(), e));
            }
        };
        let (mut ct_vertices, ct_splittings) = match ufo_model::ct_vertices(
            &ct_vertices_content,
            &ct_vertices_content,
            &couplings,
            &ident_map,
            &lorentz_structures,
        ) {
            Ok(x) => x,
            Err(e) => {
                return Err(ModelError::parse_error("CT_vertices.py", Box::new(e)));
            }
        };
        // Mark counterterm vertices with coupling order `CT`
        for v in ct_vertices.values_mut() {
            v.add_coupling("CT", 1);
        }
        vertices.extend(ct_vertices.into_iter());
        splittings.extend(ct_splittings.into_iter());
    }

    Ok(UFOModel::new(particles, vertices, coupling_orders, splittings))
}

pub fn sm() -> UFOModel {
    let (ident_map, particles): (HashMap<String, String>, IndexMap<String, UFOParticle>) =
        ufo_model::particles(SM_PARTICLES, &"")
            .unwrap()
            .into_iter()
            .map(|(py_name, p)| ((py_name, p.anti_name.clone()), (p.name.clone(), p)))
            .unzip();
    let coupling_orders = ufo_model::coupling_orders(SM_COUPLING_ORDERS, &"").unwrap();
    let couplings = ufo_model::couplings(SM_COUPLINGS, &"").unwrap();
    let lorentz_structures = ufo_model::lorentz_structures(SM_LORENTZ, &"").unwrap();
    let (vertices, _) = ufo_model::vertices(SM_VERTICES, &"", &couplings, &ident_map, &lorentz_structures).unwrap();
    UFOModel::new(particles, vertices, coupling_orders, HashMap::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;
    use test_log::test;
    use util::{HashMap, IndexMap};

    #[test]
    fn peg_ufo_parse_test() {
        let path = PathBuf::from("../../tests/models/QCD_UFO");
        let model = parse_ufo_model(&path).unwrap();
        let model_ref = UFOModel::new(
            IndexMap::from_iter([
                (
                    String::from("u"),
                    UFOParticle::new("u", "u~", 1, 3, 9000001, "u", "\\bar{u}", LineStyle::Straight, true),
                ),
                (
                    String::from("u~"),
                    UFOParticle::new("u~", "u", -1, -3, -9000001, "\\bar{u}", "u", LineStyle::Straight, true),
                ),
                (
                    String::from("c"),
                    UFOParticle::new("c", "c~", 1, 3, 9000002, "c", "\\bar{c}", LineStyle::Straight, true),
                ),
                (
                    String::from("c~"),
                    UFOParticle::new("c~", "c", -1, -3, -9000002, "\\bar{c}", "c", LineStyle::Straight, true),
                ),
                (
                    String::from("t"),
                    UFOParticle::new("t", "t~", 1, 3, 9000003, "t", "\\bar{t}", LineStyle::Straight, true),
                ),
                (
                    String::from("t~"),
                    UFOParticle::new("t~", "t", -1, -3, -9000003, "\\bar{t}", "t", LineStyle::Straight, true),
                ),
                (
                    String::from("G"),
                    UFOParticle::new("G", "G", 2, 8, 9000004, "G", "G", LineStyle::Curly, false),
                ),
            ]),
            IndexMap::from_iter([
                (
                    "V_1".to_string(),
                    UFOVertex::new(
                        "V_1".to_string(),
                        vec!["G".to_string(); 3],
                        vec![],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_2".to_string(),
                    UFOVertex::new(
                        "V_2".to_string(),
                        vec!["G".to_string(); 4],
                        vec![],
                        HashMap::from_iter([("QCD".to_string(), 2)]),
                    ),
                ),
                (
                    "V_3".to_string(),
                    UFOVertex::new(
                        "V_3".to_string(),
                        vec!["u".to_string(), "u~".to_string(), "G".to_string()],
                        vec![1, 0],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_4".to_string(),
                    UFOVertex::new(
                        "V_4".to_string(),
                        vec!["c".to_string(), "c~".to_string(), "G".to_string()],
                        vec![1, 0],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_5".to_string(),
                    UFOVertex::new(
                        "V_5".to_string(),
                        vec!["t".to_string(), "t~".to_string(), "G".to_string()],
                        vec![1, 0],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
            ]),
            vec!["QCD".to_string()],
            HashMap::default(),
        );
        assert_eq!(model, model_ref);
    }

    #[test]
    fn ufo_sm_test() {
        let path = PathBuf::from("../../tests/models/Standard_Model_UFO");
        let model = parse_ufo_model(&path);
        assert!(model.is_ok());
    }

    #[test]
    fn ufo_qcd_4f_test() {
        let path = PathBuf::from("../../tests/models/QCD_4F_UFO");
        let model = parse_ufo_model(&path);
        assert!(model.is_ok());
    }

    #[test]
    fn ufo_lorentz_test() {
        let structure = "P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)";
        let res = ufo_model::lorentz_structure(structure).unwrap();
        assert_eq!(res, vec![]);
    }

    #[test]
    fn ufo_lorentz_loop_test() {
        let structure = r"FF = Lorentz(name = 'FF',
                spins = [ 2, 2],
                structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - Gamma(3,2,-1)*Gamma(4,-1,-2)*ProjP(-2,1)')";
        let res = ufo_model::lorentz(structure, structure);
        assert_eq!(res.unwrap(), ("FF", vec![1, 0]));
    }

    #[test]
    fn ufo_ct_test() {
        let path = PathBuf::from("../../tests/models/SMEFTFR_dphi_phiG");
        let model = parse_ufo_model(&path);
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.vertices.len(), 165);
    }
}
