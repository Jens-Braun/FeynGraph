use crate::util::{HashMap, IndexMap};
use crate::{
    model::{InteractionVertex, LineStyle, Model, ModelError, Particle, Statistic},
    util::contract_indices,
};
use itertools::Itertools;
use log;
use peg;
use std::path::Path;

const SM_PARTICLES: &str = include_str!("../../tests/resources/Standard_Model_UFO/particles.py");
const SM_COUPLING_ORDERS: &str = include_str!("../../tests/resources/Standard_Model_UFO/coupling_orders.py");
const SM_COUPLINGS: &str = include_str!("../../tests/resources/Standard_Model_UFO/couplings.py");
const SM_LORENTZ: &str = include_str!("../../tests/resources/Standard_Model_UFO/lorentz.py");
const SM_VERTICES: &str = include_str!("../../tests/resources/Standard_Model_UFO/vertices.py");

#[derive(Debug)]
enum Value<'a> {
    Int(isize),
    Rational(isize, isize),
    String(&'a str),
    Bool(bool),
    List(Vec<Value<'a>>),
    SIDict(HashMap<String, usize>),
    CODict(HashMap<(usize, usize), String>),
    Particle(Particle),
    None,
}

impl<'a> Value<'a> {
    fn int(self) -> Result<isize, &'static str> {
        match self {
            Self::Int(i) => Ok(i),
            _ => Err("Int"),
        }
    }

    fn str(self) -> Result<&'a str, &'static str> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err("String"),
        }
    }

    fn i_list(self) -> Result<Vec<isize>, &'static str> {
        match self {
            Self::List(l) => Ok(l.into_iter().map(|v| v.int()).collect::<Result<Vec<_>, &str>>()?),
            _ => Err("List of ints"),
        }
    }

    fn s_list(self) -> Result<Vec<String>, &'static str> {
        match self {
            Self::List(l) => Ok(l
                .into_iter()
                .map(|v| v.str().map(|s| s.to_owned()))
                .collect::<Result<Vec<_>, &str>>()?),
            _ => Err("List of strings"),
        }
    }

    fn si_dict(self) -> Result<HashMap<String, usize>, &'static str> {
        match self {
            Self::SIDict(d) => {
                return Ok(d);
            }
            _ => Err("String-int dict")?,
        }
    }

    fn co_dict(self) -> Result<HashMap<(usize, usize), String>, &'static str> {
        match self {
            Self::CODict(d) => {
                return Ok(d);
            }
            _ => Err("(int, int)-String dict")?,
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
        rule string() -> Value<'input> = s:$(("\"" [^ '"' ]* "\"") / ("\'" [^ '\'' ]* "\'")) {
            Value::String(&s[1..s.len()-1])
        }
        rule si_dict() -> Value<'input> =
            "{" _ entries:((key:string() _ ":" _ val:value() {(key, val)}) ** (_ "," _)) _ "}" {?
                let mut result = HashMap::default();
                for (k, v) in entries.into_iter().map(
                    |(key, val)| (key.str().unwrap().to_owned(), val.int().map(|i| i as usize))
                ) {
                    match v {
                        Ok(v) => {
                            if let Some(i) =  result.insert(k.clone(), v) {
                                log::warn!("Value '{}' appears multiple times in dict, keeping only value {}", k, i);
                            }
                        }
                        _ => Err("String-int dict")?
                    }
                }
                Ok(Value::SIDict(result))
            }

        rule co_dict() -> Value<'input> =
            "{" _ entries:(("(" _ i:int() _ "," _ j:int() ")" _ ":" _ c:("C." _ c:name() {c}) {((i, j), c)}) ** (_ "," _)) _ "}" {?
                let mut result = HashMap::default();
                for ((i, j), c) in entries.into_iter() {
                    let i = i.int()?.try_into().or(Err("Non-negative int"))?;
                    let j = j.int()?.try_into().or(Err("Non-negative int"))?;
                    if let Some(v) =  result.insert((i, j), c.to_owned()) {
                        log::warn!("Basis element '({}, {})' appears multiple times in dict, keeping only coupling {}", i, j, v);
                    }
                }
                Ok(Value::CODict(result))
            }

        rule p_list() -> Value<'input> =
            "[" _ vals:(("P." _ name:name() {Value::String(name)}) ** (_ "," _)) _ "]" {Value::List(vals)}

        rule l_list() -> Value<'input> =
            "[" _ vals:(("L." _ name:name() {Value::String(name)}) ** (_ "," _)) _ "]" {Value::List(vals)}

        rule value() -> Value<'input> = rational() / int() / bool() / string() / p_list() / l_list() / si_dict() / co_dict()
        rule property_value() -> Value<'input> =
            value()
            / "[" _ vals:(value() ** (_ "," _)) _ "]" { Value::List(vals) }
            / ([^',' | ')']* {Value::None})

        rule property() -> (&'input str, Value<'input>) = prop:name() _ "=" _ value:property_value() {(prop, value)}

        rule particle() -> (&'input str, Value<'input>) =
            pos: position!() py_name:name() _ "=" _ "Particle(" _ props:(property() **<1,> (_ "," _)) _ ")" {?
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
                                        pos,
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
                            log::warn!("Unable to determine linestyle for particle '{}' at {} from spin an color", name.unwrap(), pos);
                            LineStyle::None
                        },
                    });
                }
                let mut statistic = Statistic::Bose;
                if let Some(s) = twospin && (s < 0 || s % 2 == 1) {
                    statistic = Statistic::Fermi;
                }
                Ok((py_name, Value::Particle(Particle::new(
                    name.unwrap(),
                    antiname.unwrap(),
                    pdg_code.unwrap(),
                    texname.unwrap(),
                    antitexname.unwrap(),
                    linestyle.unwrap(),
                    statistic
                ))))
            }

        rule anti_particle() -> (&'input str, Value<'input>) =
            _ anti_ident:name() _ "=" _ p_ident:name() ".anti()" {(anti_ident, Value::String(p_ident))}

        rule coupling_order() -> String =
            pos: position!() py_name:name() _ "=" _ "CouplingOrder(" _ props:(property() **<1,> (_ "," _)) _ ")" {?
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
                    None => Err("Coupling order name"),
                    Some(name) => Ok(name)
                }
            }

        rule coupling() -> (String, HashMap<String, usize>) =
            pos: position!() py_name:name() _ "=" _ "Coupling(" _ props:(property() **<1,> (_ "," _)) _ ")" {?
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
                                pos,
                                );
                        }
                    }
                }
                match orders {
                    None => Err("Coupling order dict"),
                    Some(orders) => {
                        return Ok((py_name.to_owned(), orders));
                    }
                }
            }

        rule vertex(
            couplings: &HashMap<String, HashMap<String, usize>>,
            ident_map: &HashMap<String, String>,
            lorentz_structures: &HashMap<String, Vec<isize>>
        ) -> Vec<InteractionVertex> =
            pos: position!() py_name:name() _ "=" _ "Vertex(" _ props:(property() **<1,> (_ "," _)) _ ")" {?
                let mut particles = None;
                let mut name = None;
                let mut coupling_dict = None;
                let mut lorentz_list = None;

                for (prop, value) in props {
                    match prop.to_lowercase().as_str() {
                        "name" => name = Some(value.str()?.to_owned()),
                        "particles" => particles = Some(value.s_list()?),
                        "couplings" => coupling_dict = Some(value.co_dict()?),
                        "lorentz" => lorentz_list = Some(value.s_list()?),
                        "color" => (),
                        x => {
                            log::warn!(
                                "Option '{}' for vertex '{}' at {} is unknown, ignoring",
                                x,
                                py_name,
                                pos,
                                );
                        }
                    }
                }
                let particles = particles.unwrap().iter().map(|p| ident_map.get(p).unwrap().clone()).collect_vec();
                let name = name.unwrap();

                let mut unique_spin_mappings: Vec<(Vec<isize>, Vec<usize>)> = Vec::new();
                for (i, spin_map) in lorentz_list.unwrap().iter().map(|s| lorentz_structures.get(s).unwrap()).enumerate() {
                    match unique_spin_mappings.iter_mut().find(|(v, _)| **v == *spin_map ) {
                        None => unique_spin_mappings.push((spin_map.clone(), vec![i])),
                        Some((_, l)) => l.push(i)
                    }
                }

                #[allow(clippy::type_complexity)]
                let mut unique_coupling_orders: Vec<(&HashMap<String, usize>, Vec<(usize, usize)>)> = vec![];
                if let Some(coupling_dict) = coupling_dict {
                    for ((i, j), c) in coupling_dict.iter() {
                        match couplings.get(c) {
                            Some(d) => {
                                match unique_coupling_orders.iter_mut().find(|(v, _)| **v == *d ) {
                                    None => unique_coupling_orders.push((d, vec![(*i, *j)])),
                                    Some((_, l)) => l.push((*i, *j))
                                }
                            },
                            None => {
                                log::warn!("Vertex '{}' at '{}' contains coupling '{}', which is not specified in \
                                'couplings.py', ignoring", py_name, pos, c);
                            }
                        }
                    }
                } else {
                    log::warn!("No couplings specified for vertex {} at {}", py_name, pos);
                }

                let mut vertices = Vec::new();
                if unique_coupling_orders.len() > 1 || unique_spin_mappings.len() > 1 {
                    for (spin_map, lorentz_indices) in unique_spin_mappings.iter() {
                        if unique_coupling_orders.is_empty() {
                            vertices.push(InteractionVertex::new(
                                format!("{}_{}", &name, vertices.len()),
                                particles.clone(),
                                spin_map.clone(),
                                HashMap::default(),
                            ))
                        } else {
                            for (d, l) in unique_coupling_orders.iter() {
                                if l.iter().any(|(_, i)| lorentz_indices.contains(i)) {
                                    vertices.push(InteractionVertex::new(
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
                    (1, 0) => {
                        vertices.push(InteractionVertex::new(
                            name,
                            particles.clone(),
                            unique_spin_mappings[0].0.clone(),
                            HashMap::default(),
                        ))
                    },
                    (1, 1) => {
                        vertices.push(InteractionVertex::new(
                            name,
                            particles.clone(),
                            unique_spin_mappings[0].0.clone(),
                            unique_coupling_orders[0].0.clone(),
                        ))
                    },
                    (1, x) => {
                        log::warn!(
                            "Ambiguous coupling powers for vertex {}, splitting into {} vertices '{}_0' .. '{}_{}'",
                            &name, x, &name, &name, x - 1
                        );
                    },
                    (x, 1) => {
                        log::warn!(
                            "Ambiguous spin mapping for vertex {}, splitting into {} vertices '{}_0' .. '{}_{}'",
                            &name, x, &name, &name, x - 1
                        );
                    },
                    (_, _) => {
                        log::warn!(
                            "Ambiguous coupling powers and spin mapping for vertex {}, \
                            splitting into {} vertices '{}_0' .. '{}_{}'",
                            &name, vertices.len(), &name, &name, vertices.len() - 1
                        );
                    }
                }

                Ok(vertices)
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

        rule lorentz() -> (&'input str, Vec<isize>) =
            pos: position!() py_name:name() _ "=" _ "Lorentz(" _ props:(property() **<1,> (_ "," _)) _ ")" {?
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
                                pos,
                                );
                        }
                    }
                }
                let mut connections = lorentz_structure(structure.unwrap()).or(Err("Lorentz structure"))?;

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
                                        containing this lorentz structure might be wrong!", name.unwrap(), pos);
                        }
                        let j = connected[0];
                        seen[i] = true;
                        seen[j] = true;
                        connections.push((i as isize, j as isize));
                    }
                }
                return Ok((py_name, contract_indices(connections)));
            }

        pub rule particles() -> Vec<(String, Particle)> =
            _ (!particle() [_])* _ particles:((anti_particle() / particle()) ** _) _ {?
                let mut res = Vec::with_capacity(particles.len());
                for (py_name, v) in particles.into_iter() {
                    match v {
                        Value::Particle(p) => {res.push((py_name.to_owned(), p))},
                        Value::String(p_ident) => {
                            let p_pos = match res.iter().position(|(ident, _)| ident == p_ident) {
                                Some(i) => i,
                                None => {
                                    return Err("Anti-particle of previously declared particle");
                                }
                            };
                            res.insert(p_pos+1, (py_name.to_owned(), res[p_pos].1.clone().into_anti()));
                        },
                        _ => unreachable!()
                    }
                }
                return Ok(res);
            }

        pub rule coupling_orders() -> Vec<String> =
        _ (!coupling_order() [_])* _ orders:(coupling_order() ** _) _ {orders}

        pub rule couplings() -> HashMap<String, HashMap<String, usize>> =
        _ (!coupling() [_])* _ couplings:(coupling() ** _) _ {couplings.into_iter().collect()}

        pub rule lorentz_structures() -> HashMap<String, Vec<isize>> =
        _ (!lorentz() [_])* _ structures:(lorentz() ** _) _ {
            structures.into_iter().map(|(s, v)| (s.to_owned(), v)).collect()
        }

        pub rule vertices(
            couplings: &HashMap<String, HashMap<String, usize>>,
            ident_map: &HashMap<String, String>,
            lorentz_structures: &HashMap<String, Vec<isize>>
        ) -> IndexMap<String, InteractionVertex> =
        _ (!vertex(couplings, ident_map, lorentz_structures) [_])* _
        vertices:(vertex(couplings, ident_map, lorentz_structures) ** _) _ {
                vertices.into_iter().flatten().map(|v| (v.name.clone(), v)).collect()
            }
    }
}

pub fn parse_ufo_model(path: &Path) -> Result<Model, ModelError> {
    let particle_content = match std::fs::read_to_string(path.join("particles.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::IOError(
                path.join("particles.py").to_str().unwrap().to_owned(),
                e,
            ));
        }
    };
    let (ident_map, particles): (HashMap<String, String>, IndexMap<String, Particle>) =
        match ufo_model::particles(&particle_content) {
            Ok(x) => x
                .into_iter()
                .map(|(py_name, p)| ((py_name, p.name.clone()), (p.name.clone(), p)))
                .unzip(),
            Err(e) => {
                return Err(ModelError::ParseError("particles.py".into(), e));
            }
        };

    let coupling_order_content = match std::fs::read_to_string(path.join("coupling_orders.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::IOError(
                path.join("coupling_orders.py").to_str().unwrap().to_owned(),
                e,
            ));
        }
    };
    let mut coupling_orders = match ufo_model::coupling_orders(&coupling_order_content) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::ParseError("coupling_orders.py".into(), e));
        }
    };

    let coupling_content = match std::fs::read_to_string(path.join("couplings.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::IOError(
                path.join("couplings.py").to_str().unwrap().to_owned(),
                e,
            ));
        }
    };
    let couplings = match ufo_model::couplings(&coupling_content) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::ParseError("couplings.py".into(), e));
        }
    };

    let lorentz_content = match std::fs::read_to_string(path.join("lorentz.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::IOError(
                path.join("lorentz.py").to_str().unwrap().to_owned(),
                e,
            ));
        }
    };
    let lorentz_structures = match ufo_model::lorentz_structures(&lorentz_content) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::ParseError("lorentz.py".into(), e));
        }
    };

    let vertices_content = match std::fs::read_to_string(path.join("vertices.py")) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::IOError(
                path.join("vertices.py").to_str().unwrap().to_owned(),
                e,
            ));
        }
    };
    let vertices = match ufo_model::vertices(&vertices_content, &couplings, &ident_map, &lorentz_structures) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::ParseError("vertices.py".into(), e));
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

    Ok(Model::new(particles, vertices, coupling_orders))
}

pub fn sm() -> Model {
    let (ident_map, particles): (HashMap<String, String>, IndexMap<String, Particle>) =
        ufo_model::particles(SM_PARTICLES)
            .unwrap()
            .into_iter()
            .map(|(py_name, p)| ((py_name, p.name.clone()), (p.name.clone(), p)))
            .unzip();
    let coupling_orders = ufo_model::coupling_orders(SM_COUPLING_ORDERS).unwrap();
    let couplings = ufo_model::couplings(SM_COUPLINGS).unwrap();
    let lorentz_structures = ufo_model::lorentz_structures(SM_LORENTZ).unwrap();
    let vertices = ufo_model::vertices(SM_VERTICES, &couplings, &ident_map, &lorentz_structures).unwrap();
    Model::new(particles, vertices, coupling_orders)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{InteractionVertex, LineStyle, Model, Particle, Statistic};
    use crate::util::{HashMap, IndexMap};
    use std::path::PathBuf;
    use test_log::test;

    #[test]
    fn peg_ufo_parse_test() {
        let path = PathBuf::from("tests/resources/QCD_UFO");
        let model = parse_ufo_model(&path).unwrap();
        let model_ref = Model::new(
            IndexMap::from_iter([
                (
                    String::from("u"),
                    Particle::new("u", "u~", 9000001, "u", "u~", LineStyle::Straight, Statistic::Fermi),
                ),
                (
                    String::from("u~"),
                    Particle::new("u~", "u", -9000001, "u~", "u", LineStyle::Straight, Statistic::Fermi),
                ),
                (
                    String::from("c"),
                    Particle::new("c", "c~", 9000002, "c", "c~", LineStyle::Straight, Statistic::Fermi),
                ),
                (
                    String::from("c~"),
                    Particle::new("c~", "c", -9000002, "c~", "c", LineStyle::Straight, Statistic::Fermi),
                ),
                (
                    String::from("t"),
                    Particle::new("t", "t~", 9000003, "t", "t~", LineStyle::Straight, Statistic::Fermi),
                ),
                (
                    String::from("t~"),
                    Particle::new("t~", "t", -9000003, "t~", "t", LineStyle::Straight, Statistic::Fermi),
                ),
                (
                    String::from("G"),
                    Particle::new("G", "G", 9000004, "G", "G", LineStyle::Curly, Statistic::Bose),
                ),
            ]),
            IndexMap::from_iter([
                (
                    "V_1".to_string(),
                    InteractionVertex::new(
                        "V_1".to_string(),
                        vec!["G".to_string(); 3],
                        vec![],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_2".to_string(),
                    InteractionVertex::new(
                        "V_2".to_string(),
                        vec!["G".to_string(); 4],
                        vec![],
                        HashMap::from_iter([("QCD".to_string(), 2)]),
                    ),
                ),
                (
                    "V_3".to_string(),
                    InteractionVertex::new(
                        "V_3".to_string(),
                        vec!["u~".to_string(), "u".to_string(), "G".to_string()],
                        vec![1, 0],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_4".to_string(),
                    InteractionVertex::new(
                        "V_4".to_string(),
                        vec!["c~".to_string(), "c".to_string(), "G".to_string()],
                        vec![1, 0],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_5".to_string(),
                    InteractionVertex::new(
                        "V_5".to_string(),
                        vec!["t~".to_string(), "t".to_string(), "G".to_string()],
                        vec![1, 0],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
            ]),
            vec!["QCD".to_string()],
        );
        assert_eq!(model, model_ref);
    }

    #[test]
    fn ufo_sm_test() {
        let path = PathBuf::from("tests/resources/Standard_Model_UFO");
        let model = parse_ufo_model(&path);
        assert!(model.is_ok());
    }

    #[test]
    fn ufo_qcd_4f_test() {
        let path = PathBuf::from("tests/resources/QCD_4F_UFO");
        let model = parse_ufo_model(&path);
        assert!(model.is_ok());
    }

    #[test]
    fn ufo_lorentz_test() {
        let structure = "P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)";
        let res = ufo_model::lorentz_structure(structure).unwrap();
        assert_eq!(res, vec![]);
    }
}
