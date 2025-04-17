use std::{collections::HashMap, path::Path};
use peg;
use log;
use itertools::Itertools;
use indexmap::IndexMap;
use crate::model::{
    ModelError, Model, Particle, InteractionVertex, LineStyle, Statistic
};

const SM_PARTICLES: &str = include_str!("../../tests/resources/Standard_Model_UFO/particles.py");
const SM_COUPLING_ORDERS: &str = include_str!("../../tests/resources/Standard_Model_UFO/coupling_orders.py");
const SM_COUPLINGS: &str = include_str!("../../tests/resources/Standard_Model_UFO/couplings.py");
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
    None
}

impl<'a> Value<'a> {
    fn int(self) -> Result<isize, &'static str> {
        match self {
            Self::Int(i) => Ok(i),
            _ => Err("Int")
        }
    }

    fn str(self) -> Result<&'a str, &'static str> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err("String")
        }
    }

    fn s_list(self) -> Result<Vec<String>, &'static str> {
        match self {
            Self::List(l) => {
                Ok(l.into_iter().map(|v| v.str().map(|s| s.to_owned())).collect::<Result<Vec<_>, &str>>()?)
            },
            _ => Err("List of strings")
        }
    }

    fn si_dict(self) -> Result<HashMap<String, usize>, &'static str> {
        match self {
            Self::SIDict(d) => {
                return Ok(d);
            }
            _ => Err("String-int dict")?
        }
    }

    fn co_dict(self) -> Result<HashMap<(usize, usize), String>, &'static str> {
        match self {
            Self::CODict(d) => {
                return Ok(d);
            }
            _ => Err("(int, int)-String dict")?
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
                let mut result = HashMap::new();
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
                let mut result = HashMap::new();
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
                                        "Option 'linestyle' for particle '{}' at {} has unknown value '{}', defaulting to 'dashed'",
                                        py_name,
                                        pos,
                                        x
                                        );
                                    LineStyle::Dashed
                                }
                            })
                        },
                        _ => ()
                    }
                }
                if linestyle.is_none() {
                    linestyle = Some(match (twospin, color) {
                        (Some(0), _) => LineStyle::Dashed,
                        (Some(1), Some(1)) => LineStyle::Swavy,
                        (Some(1), _) if *name.unwrap() != *antiname.unwrap() => LineStyle::Straight,
                        (Some(1), _) => LineStyle::Scurly,
                        (Some(2), Some(1)) => LineStyle::Wavy,
                        (Some(2), _) => LineStyle::Curly,
                        (Some(4), _) => LineStyle::Double,
                        (Some(-2), _) => LineStyle::Dotted,
                        _ => {
                            log::warn!("Unable to determine linestyle for particle '{}' at {} from spin an color, using default 'dashed'", name.unwrap(), pos);
                            LineStyle::Dashed
                        },
                    });
                }
                let mut statistic = Statistic::Bose;
                if let Some(s) = twospin {
                    if s < 0 || s % 2 == 1 {
                        statistic = Statistic::Fermi;
                    }
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
                couplings: &HashMap<String, HashMap<String, usize>>, ident_map: &HashMap<String, String>
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
                    let vertices;
                    #[allow(clippy::type_complexity)]
                    let mut unique_coupling_orders: Vec<(&HashMap<String, usize>, Vec<(usize, usize)>)>;
                    if let Some(coupling_dict) = coupling_dict {
                        unique_coupling_orders = Vec::new();
                        for ((i, j), c) in coupling_dict.iter() {
                            match couplings.get(c) {
                                Some(d) => {
                                    match unique_coupling_orders.iter_mut().find(|(v, _)| **v == *d ) {
                                        None => unique_coupling_orders.push((d, vec![(*i, *j)])),
                                        Some((_, l)) => l.push((*i, *j))
                                    }
                                },
                                None => {
                                    log::warn!("Vertex '{}' at '{}' contains coupling '{}', which is not specified in 'couplings.py', ignoring", py_name, pos, c);
                                }
                            }
                        }
                        if unique_coupling_orders.len() > 1 {
                            vertices = unique_coupling_orders.into_iter().enumerate().map(
                                |(i, (d, _l))| InteractionVertex {
                                    particles: particles.clone(),
                                    name: format!("{}_{}", name.as_ref().unwrap(), i),
                                    coupling_orders: d.clone()
                                }
                            ).collect_vec();
                            log::warn!("Vertex '{}' at {} has ambiguous coupling powers, the vertex will be split as\n{:#?}", py_name, pos, &vertices);
                        } else {
                            vertices = vec![InteractionVertex {
                                    particles,
                                    name: name.unwrap(),
                                    coupling_orders: unique_coupling_orders[0].0.clone()
                                }]
                        }
                    } else {
                        log::warn!("No couplings specified for vertex {} at {}", py_name, pos);
                        vertices = vec![InteractionVertex {
                            particles,
                            name: name.unwrap(),
                            coupling_orders: HashMap::new()
                        }]
                    }
                    Ok(vertices)
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

            pub rule vertices(
                couplings: &HashMap<String, HashMap<String, usize>>, ident_map: &HashMap<String, String>
            ) -> IndexMap<String, InteractionVertex> =
            _ (!vertex(couplings, ident_map) [_])* _ vertices:(vertex(couplings, ident_map) ** _) _ {
                    vertices.into_iter().flatten().map(|v| (v.name.clone(), v)).collect()
                }
    }
}

pub fn parse_ufo_model(path: &Path) -> Result<Model, ModelError> {
    let particle_content = match std::fs::read_to_string(path.join("particles.py")) {
        Ok(x) => x,
        Err(e) => { return Err(ModelError::IOError(path.join("particles.py").to_str().unwrap().to_owned(), e)); }
    };
    let (ident_map, particles): (HashMap<String, String>, IndexMap<String, Particle>) 
        = match ufo_model::particles(&particle_content) {
            Ok(x) => { 
                x.into_iter().map(
                    |(py_name, p)| ((py_name, p.name.clone()), (p.name.clone(), p))
                ).unzip()
            },
            Err(e) => { return Err(ModelError::ParseError("particles.py".into(), e)); }
        };

    let coupling_order_content = match std::fs::read_to_string(path.join("coupling_orders.py")) {
        Ok(x) => x,
        Err(e) => { return Err(ModelError::IOError(path.join("coupling_orders.py").to_str().unwrap().to_owned(), e)); }
    };
    let mut coupling_orders = match ufo_model::coupling_orders(&coupling_order_content) {
        Ok(x) => x,
        Err(e) => { return Err(ModelError::ParseError("coupling_orders.py".into(), e)); }
    };

    let coupling_content = match std::fs::read_to_string(path.join("couplings.py")) {
        Ok(x) => x,
        Err(e) => { return Err(ModelError::IOError(path.join("couplings.py").to_str().unwrap().to_owned(), e)); }
    };
    let couplings = match ufo_model::couplings(&coupling_content) {
        Ok(x) => x,
        Err(e) => { return Err(ModelError::ParseError("couplings.py".into(), e)); }
    };

    let vertices_content = match std::fs::read_to_string(path.join("vertices.py")) {
        Ok(x) => x,
        Err(e) => { return Err(ModelError::IOError(path.join("vertices.py").to_str().unwrap().to_owned(), e)); }
    };
    let vertices = match ufo_model::vertices(&vertices_content, &couplings, &ident_map) {
        Ok(x) => x,
        Err(e) => { return Err(ModelError::ParseError("vertices.py".into(), e)); }
    };

    for v in vertices.values() {
        for coupling in v.coupling_orders.keys() {
            if !coupling_orders.contains(coupling) {
                log::warn!("Vertex '{}' contains coupling '{}', which is not specified in 'coupling_orders.py'. Adding it now.", &v.name, coupling);
                coupling_orders.push(coupling.clone());
            }
        }
    }

    Ok(Model {
        particles,
        vertices,
        couplings: coupling_orders
    })
}

pub fn sm() -> Model {
    let (ident_map, particles): (HashMap<String, String>, IndexMap<String, Particle>) 
        = ufo_model::particles(SM_PARTICLES).unwrap().into_iter().map(
            |(py_name, p)| ((py_name, p.name.clone()), (p.name.clone(), p))
        ).unzip();
    let coupling_orders = ufo_model::coupling_orders(SM_COUPLING_ORDERS).unwrap();
    let couplings = ufo_model::couplings(SM_COUPLINGS).unwrap();
    let vertices = ufo_model::vertices(SM_VERTICES, &couplings, &ident_map).unwrap();
    Model {
        particles,
        vertices,
        couplings: coupling_orders
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use indexmap::IndexMap;
    use crate::model::{Model, Particle, InteractionVertex, LineStyle, Statistic};
    use super::*;

    #[test]
    fn peg_ufo_parse_test() {
        let path = PathBuf::from("tests/resources/QCD_UFO");
        let model = parse_ufo_model(&path).unwrap();
        let model_ref = Model {
            particles: IndexMap::from([
                (String::from("u"), Particle::new(
                    "u", "u~", 9000001, "u", "u~", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("u~"), Particle::new(
                    "u~", "u", -9000001, "u~", "u", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("c"), Particle::new(
                    "c", "c~", 9000002, "c", "c~", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("c~"), Particle::new(
                    "c~", "c", -9000002, "c~", "c", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("t"), Particle::new(
                    "t", "t~", 9000003, "t", "t~", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("t~"), Particle::new(
                    "t~", "t", -9000003, "t~", "t", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("G"), Particle::new(
                    "G", "G", 9000004, "G", "G", LineStyle::Curly, Statistic::Bose
                )),

            ]),
            vertices: IndexMap::from([
                ("V_1".to_string(), InteractionVertex {
                    name: "V_1".to_string(),
                    particles: vec!["G".to_string(); 3],
                    coupling_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_2".to_string(), InteractionVertex {
                    name: "V_2".to_string(),
                    particles: vec!["G".to_string(); 4],
                    coupling_orders: HashMap::from([("QCD".to_string(), 2)])
                }), 
                ("V_3".to_string(), InteractionVertex {
                    name: "V_3".to_string(),
                    particles: vec!["u~".to_string(), "u".to_string(), "G".to_string()],
                    coupling_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_4".to_string(), InteractionVertex {
                    name: "V_4".to_string(),
                    particles: vec!["c~".to_string(), "c".to_string(), "G".to_string()],
                    coupling_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_5".to_string(), InteractionVertex {
                    name: "V_5".to_string(),
                    particles: vec!["t~".to_string(), "t".to_string(), "G".to_string()],
                    coupling_orders: HashMap::from([("QCD".to_string(), 1)])
                })
            ]),
            couplings: vec!["QCD".to_string()],
        };
        println!("{:#?}", model_ref);
        assert_eq!(model, model_ref);
    }
}