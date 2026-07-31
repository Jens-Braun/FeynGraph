use crate::UFOModel;
use crate::components::{
    CTParameter, CTVertex, CTVertexType, ColorAtom, Coupling, CouplingOrder, CouplingValue, Lorentz, LorentzExpression,
    Parameter, ParameterValue, Particle, Rank, SpinMap, Vertex,
};
use crate::error::{ParseError, ReadError};
use crate::parser::color::ColorParser;
use crate::parser::coupling::RawCouplingValue;
use crate::parser::diagnostic::{
    AmbiguousCouplingOrder, AmbiguousLorentzRank, AmbiguousLorentzSpinMap, AmbiguousVertexRank, AmbiguousVertexSpinMap,
};
use crate::parser::lorentz::LorentzParser;
use crate::parser::parameter::ParameterParser;
use expression::Expression;
use log::warn;
pub(crate) use parameter::RawParameterValue;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use util::{HashMap, IndexMap, parser::Diagnostic, unicode::index_super};

pub mod color;
pub mod coupling;
pub mod coupling_orders;
//pub mod lexer;
mod common;
pub mod diagnostic;
pub(crate) mod expression_ast;
pub mod lorentz;
mod math_base;
mod parameter;

pub trait InnerReader {
    fn new(model: &Path) -> Result<Self, ReadError>
    where
        Self: Sized;
    fn read_particles(&self, base: &mut UFOReader) -> Result<(), ReadError>;
    fn read_vertices(&self, base: &mut UFOReader) -> Result<(), ReadError>;
    fn read_couplings(&self, base: &mut UFOReader) -> Result<(), ReadError>;
    fn read_lorentz(&self, base: &mut UFOReader) -> Result<(), ReadError>;
    fn read_parameters(&self, base: &mut UFOReader) -> Result<(), ReadError>;
    fn read_orders(&self, base: &mut UFOReader) -> Result<(), ReadError>;
    fn read_ct_parameters(&self, base: &mut UFOReader) -> Result<(), ReadError>;
    fn read_ct_vertices(&self, base: &mut UFOReader) -> Result<(), ReadError>;
}

#[derive(Default, Debug)]
pub struct ReaderConfig {
    path: PathBuf,
    ambiguity_config: AmbiguityConfig,
}

#[derive(Debug)]
pub struct AmbiguityConfig {
    unique_rank: bool,
    unique_spin_map: bool,
    unique_coupling_order: bool,
}

impl Default for AmbiguityConfig {
    fn default() -> Self {
        Self {
            unique_rank: true,
            unique_spin_map: true,
            unique_coupling_order: true,
        }
    }
}

pub struct UFOReader<'a> {
    conf: ReaderConfig,

    pub(crate) particles: IndexMap<String, Particle>,
    pub(crate) vertices: IndexMap<String, Vertex>,
    pub(crate) couplings: HashMap<String, Coupling>,
    pub(crate) lorentz: HashMap<String, Lorentz>,
    pub(crate) parameters: HashMap<String, Parameter>,
    pub(crate) orders: HashMap<String, CouplingOrder>,
    pub(crate) ct_parameters: HashMap<String, CTParameter>,
    pub(crate) ct_vertices: IndexMap<String, CTVertex>,

    lorentz_parser: LorentzParser<'a>,
    color_parser: ColorParser<'a>,
    parameter_parser: ParameterParser<'a>,
    errors: Vec<ParseError>,
    warnings: Vec<ParseError>,
    success: bool,
}

impl<'a> UFOReader<'a> {
    pub fn new(model: &Path) -> Self {
        return Self {
            conf: ReaderConfig {
                path: model.into(),
                ..Default::default()
            },

            particles: IndexMap::default(),
            vertices: IndexMap::default(),
            couplings: HashMap::default(),
            lorentz: HashMap::default(),
            parameters: HashMap::default(),
            orders: HashMap::default(),
            ct_parameters: HashMap::default(),
            ct_vertices: IndexMap::default(),

            lorentz_parser: LorentzParser::new(),
            color_parser: ColorParser::new(),
            parameter_parser: ParameterParser::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            success: true,
        };
    }

    pub fn read(&mut self) -> Result<UFOModel, ReadError> {
        #[cfg(feature = "py-reader")]
        return self.read_with::<crate::python::cpython_reader::CPythonReader>();
        #[cfg(not(feature = "py-reader"))]
        todo!()
    }

    pub fn emit(&self) -> std::io::Result<()> {
        for e in self.errors.iter() {
            e.emit()?;
        }
        Ok(())
    }

    pub fn emit_warnings(&self) -> std::io::Result<()> {
        for e in self.warnings.iter() {
            e.emit()?;
        }
        Ok(())
    }

    pub fn split_rank(&mut self) -> &mut Self {
        self.conf.ambiguity_config.unique_rank = true;
        self
    }

    pub fn split_map(&mut self) -> &mut Self {
        self.conf.ambiguity_config.unique_spin_map = true;
        self
    }

    pub fn split_coupling_order(&mut self) -> &mut Self {
        self.conf.ambiguity_config.unique_coupling_order = true;
        self
    }

    pub fn read_with<I: InnerReader>(&mut self) -> Result<UFOModel, ReadError> {
        let inner = I::new(&self.conf.path)?;
        inner.read_particles(self)?;
        inner.read_couplings(self)?;
        inner.read_lorentz(self)?;
        inner.read_parameters(self)?;
        inner.read_orders(self)?;
        inner.read_vertices(self)?;
        inner.read_ct_parameters(self)?;
        inner.read_ct_vertices(self)?;
        let anti_map = self
            .particles
            .values()
            .map(|p| {
                self.particles
                    .iter()
                    .position(|(anti_name, _)| &p.anti_name == anti_name)
                    .unwrap()
            })
            .collect();
        if self.success {
            return Ok(UFOModel {
                particles: std::mem::take(&mut self.particles),
                vertices: std::mem::take(&mut self.vertices),
                couplings: std::mem::take(&mut self.couplings),
                lorentz: std::mem::take(&mut self.lorentz),
                parameters: std::mem::take(&mut self.parameters),
                orders: std::mem::take(&mut self.orders),
                ct_parameters: std::mem::take(&mut self.ct_parameters),
                ct_vertices: std::mem::take(&mut self.ct_vertices),
                anti_map,
            });
        } else {
            return Err(ReadError::new(
                crate::error::ReadErrorKind::InvalidModelError,
                "Unable to parse UFO model",
            ));
        }
    }

    pub(crate) fn push_particle(&mut self, p: Particle) {
        match self.particles.insert(p.name.clone(), p) {
            None => (),
            Some(p) => warn!(
                "The particle `{}` appears multiple times in the model, overwriting previous.",
                p.name
            ),
        }
    }

    pub(crate) fn push_vertex(
        &mut self,
        name: String,
        particles: Vec<String>,
        lorentz: Vec<String>,
        color: Vec<String>,
        couplings: HashMap<(usize, usize), String>,
    ) {
        let color: Vec<_> = color
            .into_iter()
            .enumerate()
            .map(|(i, c)| self.color_parser.parse(&c, &name, i))
            .collect();
        if color.iter().any(|c| c.is_err()) {
            self.errors.extend(
                color
                    .into_iter()
                    .filter_map(|c| if let Err(e) = c { Some(e) } else { None }),
            );
        } else {
            let mut vertices = Vec::new();
            if self.conf.ambiguity_config.unique_rank
                || self.conf.ambiguity_config.unique_spin_map | self.conf.ambiguity_config.unique_coupling_order
            {
                // Assign each lorentz expression an index based on its rank
                let mut rank_partition = vec![0; lorentz.len()];
                if self.conf.ambiguity_config.unique_rank {
                    let ranks = lorentz
                        .iter()
                        .map(|l| self.lorentz.get(l).unwrap().rank.clone())
                        .collect::<Vec<_>>();
                    let mut unique_ranks = ranks.clone();
                    unique_ranks.sort();
                    unique_ranks.dedup();
                    for (i, r) in ranks.into_iter().enumerate() {
                        rank_partition[i] = unique_ranks.iter().position(|x| *x == r).unwrap();
                    }
                    let mut partitioning: HashMap<usize, Vec<usize>> = HashMap::default();
                    for (i, r_index) in rank_partition.iter().enumerate() {
                        if let Some(v) = partitioning.get_mut(r_index) {
                            v.push(i);
                        } else {
                            partitioning.insert(*r_index, vec![i]);
                        }
                    }
                    if partitioning.len() > 1 {
                        let mut s = String::new();
                        let mut spans = Vec::new();
                        for (i, part) in partitioning.values().enumerate() {
                            let ((c, l), gc) = couplings.iter().find(|((_, l), _)| part.contains(l)).unwrap();
                            let start = s.chars().count();
                            write!(s, "{} * C{} * L{}", gc, index_super(*c as i64), index_super(*l as i64)).unwrap();
                            if part.len() > 1 {
                                s.push_str(" + ...");
                            }
                            spans.push((start..s.chars().count()).into());
                            if i + 1 != partitioning.len() {
                                s.push_str(" + ");
                            }
                        }
                        self.warnings.push(ParseError::new(
                            name.clone(),
                            s,
                            vec![Box::new(AmbiguousVertexRank::new(unique_ranks, spans)) as Box<dyn Diagnostic>],
                        ));
                    }
                }
                // Assign each lorentz expression an index based on its spin map
                let mut map_partition = vec![0; lorentz.len()];
                if self.conf.ambiguity_config.unique_spin_map {
                    let maps = lorentz
                        .iter()
                        .map(|l| self.lorentz.get(l).unwrap().spin_map.clone())
                        .collect::<Vec<_>>();
                    let mut unique_maps = maps.clone();
                    unique_maps.sort();
                    unique_maps.dedup();
                    for (i, r) in maps.into_iter().enumerate() {
                        map_partition[i] = unique_maps.iter().position(|x| *x == r).unwrap();
                    }
                    let mut partitioning: HashMap<usize, Vec<usize>> = HashMap::default();
                    for (i, m_index) in map_partition.iter().enumerate() {
                        if let Some(v) = partitioning.get_mut(m_index) {
                            v.push(i);
                        } else {
                            partitioning.insert(*m_index, vec![i]);
                        }
                    }
                    if partitioning.len() > 1 {
                        let mut s = String::new();
                        let mut spans = Vec::new();
                        for (i, part) in partitioning.values().enumerate() {
                            let ((c, l), gc) = couplings.iter().find(|((_, l), _)| part.contains(l)).unwrap();
                            let start = s.chars().count();
                            write!(s, "{} * C{} * L{}", gc, index_super(*c as i64), index_super(*l as i64)).unwrap();
                            if part.len() > 1 {
                                s.push_str(" + ...");
                            }
                            spans.push((start..s.chars().count()).into());
                            if i + 1 != partitioning.len() {
                                s.push_str(" + ");
                            }
                        }
                        self.warnings.push(ParseError::new(
                            name.clone(),
                            s,
                            vec![Box::new(AmbiguousVertexSpinMap::new(unique_maps, spans)) as Box<dyn Diagnostic>],
                        ));
                    }
                }

                // Partition the gauge couplings such that every coupling per partition has the same coupling map
                let mut coupling_partition: HashMap<Vec<(String, usize)>, Vec<usize>> = HashMap::default();
                if self.conf.ambiguity_config.unique_coupling_order {
                    for (i, (_, gc)) in couplings.iter().enumerate() {
                        let key = self
                            .couplings
                            .get(gc)
                            .unwrap()
                            .order
                            .clone()
                            .into_iter()
                            .collect::<Vec<_>>();
                        match coupling_partition.get_mut(&key) {
                            Some(v) => v.push(i),
                            None => {
                                coupling_partition.insert(key, vec![i]);
                            }
                        }
                    }
                    if coupling_partition.len() > 1 {
                        let mut s = String::new();
                        let mut spans = Vec::new();
                        let mut unique_orders = Vec::new();
                        for (i, (d, part)) in coupling_partition.iter().enumerate() {
                            unique_orders.push(HashMap::<String, usize>::from_iter(d.into_iter().cloned()));
                            let start = s.chars().count();
                            let ((c, l), gc) = couplings.iter().nth(part[0]).unwrap();
                            write!(s, "{} * C{} * L{}", gc, index_super(*c as i64), index_super(*l as i64)).unwrap();
                            if part.len() > 1 {
                                s.push_str(" + ...");
                            }
                            spans.push((start..s.chars().count()).into());
                            if i + 1 != coupling_partition.len() {
                                s.push_str(" + ");
                            }
                        }
                        self.warnings.push(ParseError::new(
                            name.clone(),
                            s,
                            vec![Box::new(AmbiguousCouplingOrder::new(unique_orders, spans)) as Box<dyn Diagnostic>],
                        ));
                    }
                } else {
                    coupling_partition.insert(Vec::new(), (0..couplings.len()).collect());
                }
                // Combine rank and spin map indices to partition the Lorentz expressions such that the `(r, m)` entry
                // has the indices of all Lorentz expressions with rank `r` and spin map `m`
                let mut lorentz_partitioning: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
                for (i, (r_index, m_index)) in rank_partition.iter().zip(map_partition.iter()).enumerate() {
                    if let Some(v) = lorentz_partitioning.get_mut(&(*r_index, *m_index)) {
                        v.push(i);
                    } else {
                        lorentz_partitioning.insert((*r_index, *m_index), vec![i]);
                    }
                }

                // Combine all splitting criteria
                let mut partitioning = Vec::new();
                for indices in coupling_partition.into_values() {
                    let mut tmp = vec![Vec::new(); lorentz_partitioning.len()];
                    for i in indices {
                        for (k, lor) in lorentz_partitioning.values().enumerate() {
                            if lor.contains(&i) {
                                tmp[k].push(i);
                            }
                        }
                    }
                    for v in tmp {
                        if !v.is_empty() {
                            partitioning.push(v);
                        }
                    }
                }

                for indices in partitioning.into_iter() {
                    // For each partition, reduce the global set of Color / Lorentz expressions to those needed by the
                    // partition and recalculate the `(c, l)` indices for the gauge couplings based on the new sets
                    let reduced_lorentz: Vec<String> = lorentz
                        .iter()
                        .enumerate()
                        .filter_map(|(i, l)| if indices.contains(&i) { Some(l.to_owned()) } else { None })
                        .collect();
                    let reduced_color: Vec<Expression<ColorAtom>> = couplings
                        .keys()
                        .filter_map(|(c, l)| {
                            if indices.contains(l) {
                                Some(color[*c].as_ref().unwrap().clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    let reduced_couplings: HashMap<(usize, usize), String> = couplings
                        .iter()
                        .filter_map(|((c, l), gc)| {
                            if indices.contains(l) {
                                Some((
                                    (
                                        reduced_color
                                            .iter()
                                            .position(|rc| *rc == *color[*c].as_ref().unwrap())
                                            .unwrap(),
                                        reduced_lorentz.iter().position(|rl| *rl == lorentz[*l]).unwrap(),
                                    ),
                                    gc.clone(),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect();
                    let spin_map = if let Some(l) = reduced_lorentz.first() {
                        self.lorentz.get(l).unwrap().spin_map.clone()
                    } else {
                        Vec::new()
                    };
                    let coupling_orders = if let Some(c) = reduced_couplings.values().next() {
                        self.couplings.get(c).unwrap().order.clone()
                    } else {
                        HashMap::default()
                    };
                    vertices.push(Vertex {
                        name: "".into(),
                        particles: particles.clone(),
                        color: reduced_color,
                        lorentz: reduced_lorentz,
                        couplings: reduced_couplings,

                        coupling_orders,
                        spin_map,
                    });
                }
            } else {
                let spin_map = if let Some(l) = lorentz.first() {
                    self.lorentz.get(l).unwrap().spin_map.clone()
                } else {
                    Vec::new()
                };
                let coupling_orders = if let Some(c) = couplings.values().next() {
                    self.couplings.get(c).unwrap().order.clone()
                } else {
                    HashMap::default()
                };
                vertices.push(Vertex {
                    name: "".into(),
                    particles,
                    color: color.into_iter().collect::<Result<_, _>>().unwrap(),
                    lorentz,
                    couplings,

                    coupling_orders,
                    spin_map,
                });
            }
            let n = vertices.len();
            for (i, mut v) in vertices.into_iter().enumerate() {
                let uname = if n == 1 {
                    name.clone()
                } else {
                    format!("{}_{}", name.clone(), i)
                };
                v.name = uname.clone();
                match self.vertices.insert(uname, v) {
                    None => (),
                    Some(v) => warn!(
                        "The vertex `{}` appears multiple times in the model, overwriting previous.",
                        v.name
                    ),
                }
            }
        }
    }

    pub(crate) fn push_coupling(&mut self, name: String, value: RawCouplingValue, order: HashMap<String, usize>) {
        let value = match value {
            RawCouplingValue::Finite(s) => match self.parameter_parser.parse(&s, &name) {
                Ok(expr) => CouplingValue::Finite(expr),
                Err(e) => {
                    self.errors.push(e);
                    self.success = false;
                    return;
                }
            },
            RawCouplingValue::Expansion(map) => {
                let mut success = true;
                let c = CouplingValue::Expansion(
                    map.into_iter()
                        .filter_map(|(i, s)| match self.parameter_parser.parse(&s, &name) {
                            Ok(expr) => Some((i, expr)),
                            Err(e) => {
                                self.errors.push(e);
                                self.success = false;
                                success = false;
                                None
                            }
                        })
                        .collect(),
                );
                if success {
                    c
                } else {
                    return;
                }
            }
        };
        match self.couplings.insert(name.clone(), Coupling { name, value, order }) {
            None => (),
            Some(c) => warn!(
                "The coupling `{}` appears multiple times in the model, overwriting previous.",
                c.name
            ),
        }
    }

    pub(crate) fn push_lorentz(&mut self, name: String, spins: Vec<i64>, structure: String) {
        match self.lorentz_parser.parse(&structure, &name) {
            Ok(expr) => {
                let expr = expr.normalize();
                let rank = expr.rank();
                let spin_map = expr.spin_map();
                if let Some(r) = rank
                    && let Ok(Some(s)) = spin_map
                {
                    match self.lorentz.insert(
                        name.clone(),
                        Lorentz {
                            name,
                            spins,
                            structure: expr,
                            rank: r,
                            spin_map: s,
                        },
                    ) {
                        None => (),
                        Some(l) => warn!(
                            "The Lorentz structure `{}` appears multiple times in the model, overwriting previous.",
                            l.name
                        ),
                    }
                } else {
                    if rank.is_none() {
                        let split = expr.clone().split_by(LorentzExpression::ranks);
                        let mut s = String::new();
                        let mut spans = Vec::new();
                        let n = split.expressions().len();
                        for (i, e) in split.expressions().iter().enumerate() {
                            let start = s.chars().count();
                            match e {
                                LorentzExpression::Sum(exprs) => write!(s, "{} + ...", &exprs[0]).unwrap(),
                                x => write!(s, "{}", x).unwrap(),
                            };
                            spans.push((start..s.chars().count()).into());
                            if i + 1 != n {
                                s.push_str(" + ");
                            }
                        }
                        self.errors.push(ParseError::new(
                            name.clone(),
                            s,
                            vec![
                                Box::new(AmbiguousLorentzRank::new(split.unique_keys().to_owned(), spans))
                                    as Box<dyn Diagnostic>,
                            ],
                        ));
                    }
                    if let Ok(None) = spin_map {
                        let split = expr.split_by(|e| e.spin_maps().unwrap());
                        let mut s = String::new();
                        let mut spans = Vec::new();
                        let n = split.expressions().len();
                        for (i, e) in split.expressions().iter().enumerate() {
                            let start = s.chars().count();
                            match e {
                                LorentzExpression::Sum(exprs) => write!(s, "{} + ...", &exprs[0]).unwrap(),
                                x => write!(s, "{}", x).unwrap(),
                            };
                            spans.push((start..s.chars().count()).into());
                            if i + 1 != n {
                                s.push_str(" + ");
                            }
                        }
                        self.errors.push(ParseError::new(
                            name,
                            s,
                            vec![
                                Box::new(AmbiguousLorentzSpinMap::new(split.unique_keys().to_owned(), spans))
                                    as Box<dyn Diagnostic>,
                            ],
                        ));
                    } else if let Err(mut e) = spin_map {
                        e.set_name(name);
                        self.errors.push(e);
                    }
                    self.success = false;
                }
            }
            Err(e) => {
                self.success = false;
                self.errors.push(e);
            }
        }
    }

    pub(crate) fn push_parameter(
        &mut self,
        name: String,
        value: RawParameterValue,
        texname: String,
        lha_block: Option<String>,
        lha_code: Option<Vec<usize>>,
    ) {
        let expr = match &value {
            RawParameterValue::InternalComplex(x) | RawParameterValue::InternalReal(x) => {
                match self.parameter_parser.parse(x, &name) {
                    Ok(expr) => Some(expr),
                    Err(e) => {
                        self.success = false;
                        self.errors.push(e);
                        return;
                    }
                }
            }
            _ => None,
        };
        match self.parameters.insert(
            name.clone(),
            Parameter {
                name,
                value: match value {
                    RawParameterValue::ExternalReal(x) => ParameterValue::ExternalReal(x),
                    RawParameterValue::ExternalComplex(x) => ParameterValue::ExternalComplex(x),
                    RawParameterValue::InternalReal(_) => ParameterValue::InternalReal(expr.unwrap()),
                    RawParameterValue::InternalComplex(_) => ParameterValue::InternalComplex(expr.unwrap()),
                },
                texname,
                lha_block,
                lha_code,
            },
        ) {
            None => (),
            Some(p) => warn!(
                "The parameter `{}` appears multiple times in the model, overwriting previous.",
                p.name
            ),
        }
    }

    pub(crate) fn push_coupling_order(&mut self, o: CouplingOrder) {
        match self.orders.insert(o.name.clone(), o) {
            None => (),
            Some(o) => warn!(
                "The coupling order `{}` appears multiple times in the model, overwriting previous.",
                o.name
            ),
        }
    }

    pub(crate) fn push_ct_parameter(&mut self, p: CTParameter) {
        match self.ct_parameters.insert(p.name.clone(), p) {
            None => (),
            Some(p) => warn!(
                "The counterterm parameter `{}` appears multiple times in the model, overwriting previous.",
                p.name
            ),
        }
    }

    pub(crate) fn push_ct_vertex(
        &mut self,
        name: String,
        particles: Vec<String>,
        lorentz: Vec<String>,
        color: Vec<String>,
        couplings: HashMap<(usize, usize, usize), String>,
        r#type: CTVertexType,
        loop_particles: Vec<Vec<Vec<String>>>,
    ) {
        let color: Vec<_> = color
            .into_iter()
            .enumerate()
            .map(|(i, c)| self.color_parser.parse(&c, &name, i))
            .collect();
        if color.iter().any(|c| c.is_err()) {
            self.errors.extend(
                color
                    .into_iter()
                    .filter_map(|c| if let Err(e) = c { Some(e) } else { None }),
            );
        } else {
            match self.ct_vertices.insert(
                name.clone(),
                CTVertex {
                    name,
                    particles,
                    color: color.into_iter().map(|c| c.unwrap().normalize()).collect(),
                    lorentz,
                    couplings,
                    r#type,
                    loop_particles,
                },
            ) {
                None => (),
                Some(v) => warn!(
                    "The counterterm vertex `{}` appears multiple times in the model, overwriting previous.",
                    v.name
                ),
            }
        }
    }
}
