use crate::model::{InteractionVertex, LineStyle, Model, ModelError, Particle, Statistic};
use crate::util::{HashMap, IndexMap};
use itertools::Itertools;
use log;
use peg;
use std::path::Path;

#[derive(Debug)]
enum Value<'a> {
    Int(isize),
    String(&'a str),
    List(Vec<Value<'a>>),
    None,
}

enum ModelEntry<'a> {
    Prop(Particle),
    Vert(InteractionVertex),
    Misc(&'a str),
}

peg::parser!(
    grammar qgraf_model() for str {
        rule whitespace() = quiet!{[' ' | '\t' | '\n']}
        rule comment() = quiet!{"%" [^'\n']* "\n"}

        rule _() = quiet!{(comment() / whitespace())*}

        rule alphanumeric() = quiet!{['a'..='z' | 'A'..='Z' | '0'..='9' | '_']}

        rule statistic() -> Statistic = sign:$(['+' | '-'] / "+1" / "-1") {
            match sign {
                "+" | "+1" => Statistic::Bose,
                "-" | "-1" => Statistic::Fermi,
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

        rule value() -> Value<'input> = int() / string()
        rule property_value() -> Value<'input> =
            value()
            / "(" _ vals:(value() ** (_ "," _)) _ ")" { Value::List(vals) }

        rule property() -> (&'input str, Value<'input>) = prop:name() _ "=" _ value:property_value() {(prop, value)}

        rule propagator(particle_counter: &mut usize) -> Particle =
            "[" _ name:name() _ "," _ anti_name:name() _ "," _ statistic:statistic() _ [^';' | ']']* _
            props:(";" _ props:(property()** (_ "," _))? {props} )? _ "]" {?
                let mut twospin = None;
                let mut color = None;
                if let Some(Some(props)) = props {
                    for (prop, value) in props {
                        match prop.to_lowercase().as_str() {
                            "twospin" => {
                                match value {
                                    Value::Int(i) => twospin = Some(i),
                                    Value::String(s) => twospin = Some(s.parse::<isize>().or(Err("Expected int"))?),
                                    _ => {
                                        log::error!("In particle {}: property 'twospin' only accepts int values", name);
                                        Err("Expected int")?
                                    }
                                }
                            },
                            "color" => {
                                match value {
                                    Value::Int(i) => color = Some(i),
                                    Value::String(s) => color = Some(s.parse::<isize>().or(Err("Expected int"))?),
                                    _ => {
                                        log::error!("In particle {}: property 'color' only accepts int values", name);
                                        Err("Expected int")?
                                    }
                                }
                            },
                            "mass" => (),
                            "width" => (),
                            "aux" => (),
                            "conj" => (),
                            prop => log::warn!("Ignoring unknown property {} for particle '{}' in QGRAF model", prop, name)
                        }
                    }
                }
                let linestyle = match (twospin, color) {
                    (None, None) => LineStyle::Dashed,
                    (Some(0), _) => LineStyle::Dashed,
                    (Some(1), Some(1)) => LineStyle::Swavy,
                    (Some(1), _) if *name != *anti_name => LineStyle::Straight,
                    (Some(1), _) => LineStyle::Scurly,
                    (Some(2), Some(1)) => LineStyle::Wavy,
                    (Some(2), _) => LineStyle::Curly,
                    (Some(4), _) => LineStyle::Double,
                    (Some(-2), _) => LineStyle::Dotted,
                    _ => {
                        log::warn!("Unable to determine linestyle for particle {}", name);
                        LineStyle::None
                    },
                };
                let pdg_code = *particle_counter;
                *particle_counter += 1;
                Ok(Particle::new(
                    String::from(name),
                    String::from(anti_name),
                    pdg_code as isize,
                    String::from(name),
                    String::from(anti_name),
                    linestyle,
                    statistic
                ))
            }

        rule vertex(vertex_counter: &mut usize) -> InteractionVertex =
            pos: position!() "[" _ fields:(name() **<3,> (_ "," _)) _
            couplings:(";" _ couplings:(property() ** (_ "," _))? {couplings})? _ "]" {?
                let mut coupling_map = HashMap::default();
                let mut vertex_name = format!("V_{}", vertex_counter);
                if let Some(Some(couplings)) = couplings {
                    for (coupling, value) in couplings {
                        if coupling == "VL" {
                            match value {
                                Value::String(s) => vertex_name = String::from(s),
                                Value::Int(n) => vertex_name = format!("{}", n),
                                _ => {
                                    log::error!("Vertex at {}: vertex label must be string or int", pos);
                                    Err("String or int")?
                                }
                            }
                        }
                        match value {
                            Value::Int(n) => {
                                if let Some(v) =  coupling_map.insert(String::from(coupling), n.try_into().or(Err("Non-negative int"))?) {
                                    log::warn!("Coupling '{}' appears more than once, overwriting previous value", coupling);
                                }
                            },
                            Value::String(s) => {
                                let n = s.parse::<isize>().or(Err("Expected int"))?;
                                if let Some(v) = coupling_map.insert(String::from(coupling), n.try_into().or(Err("Non-negative int"))?) {
                                    log::warn!("Coupling '{}' appears more than once, overwriting previous value", coupling);
                                }
                            }
                            _ => (),
                        }
                    }
                }
                *vertex_counter += 1;
                Ok(InteractionVertex::new(
                    vertex_name,
                    fields.iter().map(|f| String::from(*f)).collect_vec(),
                    vec![],
                    coupling_map,
                ))
            }

        rule misc() -> &'input str =
            s:$("[" [^ ']']* "]")

        pub rule qgraf_model(particle_counter: &mut usize, vertex_counter: &mut usize) -> Model =
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
                            if !p.self_anti() {
                                let anti_name = p.anti_name.clone();
                                particles.insert(anti_name, p.into_anti());
                            }
                        },
                        ModelEntry::Vert(v) => {
                            for coupling in v.coupling_orders.keys() {
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

                Model::new(
                    particles,
                    vertices,
                    couplings
                )
            }
    }
);

pub(crate) fn parse_qgraf_model(path: &Path) -> Result<Model, ModelError> {
    let content = match std::fs::read_to_string(path) {
        Ok(x) => x,
        Err(e) => {
            return Err(ModelError::IOError(path.to_str().unwrap().to_owned(), e));
        }
    };
    let mut particle_counter: usize = 1;
    let mut vertex_counter: usize = 1;
    return match qgraf_model::qgraf_model(&content, &mut particle_counter, &mut vertex_counter) {
        Ok(m) => Ok(build_spin_maps(m)),
        Err(e) => Err(ModelError::ParseError(
            path.file_name().unwrap().to_str().unwrap().to_owned(),
            e,
        )),
    };
}

fn build_spin_maps(mut model: Model) -> Model {
    for v in model.vertices.values_mut() {
        let mut fermions = v
            .particles
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if model.particles.get(s).unwrap().statistic == Statistic::Fermi {
                    Some(i)
                } else {
                    None
                }
            })
            .collect_vec();
        if !fermions.is_empty() {
            if fermions.len() > 2 {
                log::warn!(
                    "Ambiguous spin flow mapping for vertex '{}', the calculated diagram signs for diagrams \
                containing this vertex might be wrong!",
                    v.name
                );
            }
            let mut spin_map = vec![-1; v.particles.len()];
            while let Some(index) = fermions.pop() {
                let out_leg = fermions.pop().unwrap();
                spin_map[index] = out_leg as isize;
                spin_map[out_leg] = index as isize;
            }
            v.spin_map = spin_map;
        }
    }
    return model;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Statistic;
    use std::path::PathBuf;
    use test_log::test;

    #[test]
    fn qgraf_qcd_test() {
        let path = PathBuf::from("tests/resources/qcd.qgraf");
        let model = parse_qgraf_model(&path).unwrap();
        let model_ref = Model::new(
            IndexMap::from_iter([
                (
                    String::from("quark"),
                    Particle::new(
                        "quark",
                        "antiquark",
                        1,
                        "quark",
                        "antiquark",
                        LineStyle::Straight,
                        Statistic::Fermi,
                    ),
                ),
                (
                    String::from("antiquark"),
                    Particle::new(
                        "antiquark",
                        "quark",
                        -1,
                        "antiquark",
                        "quark",
                        LineStyle::Straight,
                        Statistic::Fermi,
                    ),
                ),
                (
                    String::from("gluon"),
                    Particle::new("gluon", "gluon", 2, "gluon", "gluon", LineStyle::Curly, Statistic::Bose),
                ),
                (
                    String::from("ghost"),
                    Particle::new(
                        "ghost",
                        "antighost",
                        3,
                        "ghost",
                        "antighost",
                        LineStyle::Dotted,
                        Statistic::Fermi,
                    ),
                ),
                (
                    String::from("antighost"),
                    Particle::new(
                        "antighost",
                        "ghost",
                        -3,
                        "antighost",
                        "ghost",
                        LineStyle::Dotted,
                        Statistic::Fermi,
                    ),
                ),
            ]),
            IndexMap::from_iter([
                (
                    "V_1".to_string(),
                    InteractionVertex::new(
                        "V_1".to_string(),
                        vec!["antiquark".to_string(), "quark".to_string(), "gluon".to_string()],
                        vec![1, 0, -1],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_2".to_string(),
                    InteractionVertex::new(
                        "V_2".to_string(),
                        vec!["gluon".to_string(); 3],
                        vec![],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
                (
                    "V_3".to_string(),
                    InteractionVertex::new(
                        "V_3".to_string(),
                        vec!["gluon".to_string(); 4],
                        vec![],
                        HashMap::from_iter([("QCD".to_string(), 2)]),
                    ),
                ),
                (
                    "V_4".to_string(),
                    InteractionVertex::new(
                        "V_4".to_string(),
                        vec!["antighost".to_string(), "ghost".to_string(), "gluon".to_string()],
                        vec![1, 0, -1],
                        HashMap::from_iter([("QCD".to_string(), 1)]),
                    ),
                ),
            ]),
            vec!["QCD".to_string()],
        );
        assert_eq!(model, model_ref);
    }

    #[test]
    fn qgraf_sm_test() {
        let model = parse_qgraf_model(Path::new("tests/resources/sm.qgraf"));
        match &model {
            Ok(_) => (),
            Err(e) => {
                println!("{:#?}", e);
            }
        }
        assert!(model.is_ok());
    }
}
