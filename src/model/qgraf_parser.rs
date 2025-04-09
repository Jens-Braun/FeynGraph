use std::collections::HashMap;
use std::path::{Path};
use indexmap::IndexMap;
use pest::Parser;
use pest_derive::Parser;
use crate::model::{LineStyle, Model, ModelError, Particle, InteractionVertex};
use log::warn;
use crate::model::Statistic::{Bose, Fermi};

#[derive(Parser)]
#[grammar = "model/qgraf_model.pest"]
pub(crate) struct QGRAFParser;

impl QGRAFParser {
    pub(crate) fn parse_qgraf_model(path: &Path) -> Result<Model, ModelError> {
        let mut particles: IndexMap<String, Particle> = IndexMap::new();
        let mut coupling_orders: Vec<String> = Vec::new();
        let mut vertices: IndexMap<String, InteractionVertex> = IndexMap::new();

        let file_content = std::fs::read_to_string(path)?;
        let parsed_content = QGRAFParser::parse(Rule::file, &file_content)?.next().unwrap();
        let mut particle_counter = 1;
        let mut vertex_counter = 1;
        for rule in parsed_content.into_inner() {
            match rule.as_rule() {
                Rule::EOI => (),
                Rule::propagator => {
                    let mut inner_rule = rule.into_inner();
                    let field_name = inner_rule.next().unwrap().as_str().to_string();
                    let anti_name = inner_rule.next().unwrap().as_str().to_string();
                    let sign = inner_rule.next().unwrap().as_str().to_string();
                    let statistic = match sign.as_str() {
                        "+" | "+1" => Bose,
                        "-" | "-1" => Fermi,
                        _ => unreachable!()
                    };
                    let mut twospin = None;
                    let mut color = None;
                    for property in inner_rule {
                        let mut inner_property = property.into_inner();
                        let property_name = inner_property.next().unwrap().as_str().to_string();
                        match property_name.to_lowercase().as_str() {
                            "twospin" => twospin = Some(inner_property.next().unwrap().as_str().trim_matches(['\"', '\'']).parse::<i8>().unwrap()),
                            "color" => color = Some(inner_property.next().unwrap().as_str().trim_matches(['\"', '\'']).parse::<u8>().unwrap()),
                            "mass" => (),
                            "width" => (),
                            "aux" => (),
                            "conj" => (),
                            prop => {
                                warn!("Encountered unknown property '{}' in QGRAF model, ignoring", prop);
                            }
                        }
                    }
                    let linestyle;
                    if twospin.is_none() && color.is_none() {
                        linestyle = LineStyle::Dashed;
                    } else {
                        linestyle = match twospin.unwrap() {
                            0 => LineStyle::Dashed,
                            1 => {
                                if field_name != anti_name {
                                    LineStyle::Straight
                                } else if color == Some(1) {
                                    LineStyle::Swavy
                                } else {
                                    LineStyle::Scurly
                                }
                            },
                            2 => {
                                if color == Some(1) {
                                    LineStyle::Wavy
                                } else {
                                    LineStyle::Curly
                                }
                            }
                            4 => LineStyle::Double,
                            -2 => LineStyle::Dotted,
                            _ => {
                                warn!("Found spin '{}' for particle {} in model {:#?}, \
                                for which 'linestyle' is undefined. Defaulting to 'dashed'.", &twospin.unwrap(), &field_name, path);
                                LineStyle::Dashed
                            }
                        };
                    }
                    let pdg_code = particle_counter;
                    particle_counter += 1;
                    let particle = Particle::new(
                        field_name.clone(),
                        anti_name.clone(),
                        pdg_code,
                        field_name.clone(),
                        anti_name.clone(),
                        linestyle,
                        statistic,
                    );
                    particles.insert(field_name.clone(), particle.clone());
                    if !(field_name == anti_name) {
                        let anti_particle = particle.into_anti();
                        particles.insert(anti_name, anti_particle);
                    }
                },
                Rule::vertex => {
                    let inner_rule = rule.into_inner();
                    let mut vertex_particles: Vec<String> = Vec::new();
                    let mut vertex_couplings: HashMap<String, usize> = HashMap::new();
                    let mut vertex_name = None;
                    for property in inner_rule {
                        match property.as_rule() {
                            Rule::name => vertex_particles.push(property.as_str().to_string()),
                            Rule::property => {
                                let mut inner_property = property.into_inner();
                                let property_name = inner_property.next().unwrap().as_str().to_string();
                                let property_value = inner_property.next().unwrap().as_str().trim_matches(['\"', '\'']);
                                if property_name == "VL" {
                                    vertex_name = Some(property_value.to_string());
                                    continue;
                                }
                                let coupling_order = property_value.parse::<usize>().unwrap();
                                if !coupling_orders.contains(&property_name) {
                                    coupling_orders.push(property_name.clone());
                                }
                                vertex_couplings.insert(property_name.clone(), coupling_order);
                            },
                            _ => unreachable!()
                        }
                    }
                    if let None = vertex_name {
                        vertex_name = Some(format!("V_{vertex_counter}"))
                    }
                    vertices.insert(vertex_name.clone().unwrap(),
                        InteractionVertex {
                            name: vertex_name.unwrap(),
                            particles: vertex_particles,
                            couplings_orders: vertex_couplings
                        }
                    );
                    vertex_counter += 1;
                }
                Rule::misc_statement => (),
                _ => unreachable!()
            }
        }

        return Ok(Model {
            particles,
            vertices,
            coupling_orders
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use crate::model::Statistic;
    use super::*;

    #[test]
    fn qgraf_qcd_test() {
        let path = PathBuf::from("tests/resources/qcd.qgraf");
        let model = QGRAFParser::parse_qgraf_model(&path).unwrap();
        println!("{:#?}", model);
        let model_ref = Model {
            particles: IndexMap::from([
                (String::from("quark"), Particle::new(
                    "quark", "antiquark", 1, "quark", "antiquark", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("antiquark"), Particle::new(
                    "antiquark", "quark", -1, "antiquark", "quark", LineStyle::Straight, Statistic::Fermi
                )),
                (String::from("gluon"), Particle::new(
                    "gluon", "gluon", 2, "gluon", "gluon", LineStyle::Curly, Statistic::Bose
                )),
                (String::from("ghost"), Particle::new(
                    "ghost", "antighost", 3, "ghost", "antighost", LineStyle::Dotted, Statistic::Fermi
                )),
                (String::from("antighost"), Particle::new(
                    "antighost", "ghost", -3, "antighost", "ghost", LineStyle::Dotted, Statistic::Fermi
                )),

            ]),
            vertices: IndexMap::from([
                ("V_1".to_string(), InteractionVertex {
                    name: "V_1".to_string(),
                    particles: vec!["antiquark".to_string(), "quark".to_string(), "gluon".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_2".to_string(), InteractionVertex {
                    name: "V_2".to_string(),
                    particles: vec!["gluon".to_string(); 3],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_3".to_string(), InteractionVertex {
                    name: "V_3".to_string(),
                    particles: vec!["gluon".to_string(); 4],
                    couplings_orders: HashMap::from([("QCD".to_string(), 2)])
                }),
                ("V_4".to_string(), InteractionVertex {
                    name: "V_4".to_string(),
                    particles: vec!["antighost".to_string(), "ghost".to_string(), "gluon".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
            ]),
            coupling_orders: vec!["QCD".to_string()],
        };
        assert_eq!(model, model_ref);
    }
}