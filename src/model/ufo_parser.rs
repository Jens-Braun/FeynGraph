use std::collections::HashMap;
use std::path::{Path};
use indexmap::IndexMap;
use pest::Parser;
use pest_derive::Parser;
use crate::model::{LineStyle, Model, ModelError, Particle, InteractionVertex};
use log::warn;
use crate::model::ModelError::{ContentError};
use crate::model::Statistic::{Bose, Fermi};

#[derive(Parser)]
#[grammar = "model/ufo_model.pest"]
pub(crate) struct UFOParser;

impl UFOParser {
    fn parse_particles(path: &Path) -> Result<(IndexMap<String, Particle>, HashMap<String, String>), ModelError> {
        let mut particles: IndexMap<String, Particle> = IndexMap::new();
        let mut particle_id_map: HashMap<String, String> = HashMap::new();
        let particle_py_content = std::fs::read_to_string(path.join("particles.py"))?;
        let parsed_content = UFOParser::parse(Rule::particles_py, &particle_py_content)?.next().unwrap();
        for particle_rule in parsed_content.into_inner() {
            match particle_rule.as_rule() {
                Rule::particle => {
                    let position = particle_rule.line_col();
                    let mut properties = particle_rule.into_inner();
                    let particle_id = properties.next().unwrap().as_str().trim_matches(['\"', '\'']).to_string();
                    let mut pdg_code = None;
                    let mut name = None;
                    let mut antiname = None;
                    let mut texname = None;
                    let mut antitexname = None;
                    let mut linestyle = None;
                    let mut twospin = None;
                    let mut color = None;
                    let mut statistic = Bose;
                    for property in properties {
                        match property.as_rule() {
                            Rule::property_name => { name = Some(property.into_inner().next().unwrap().as_str().trim_matches(['\"', '\''])) },
                            Rule::property_antiname=> { antiname = Some(property.into_inner().next().unwrap().as_str().trim_matches(['\"', '\''])) },
                            Rule::property_pdg_code => {
                                pdg_code = Some(property.into_inner().next().unwrap().as_str().parse::<isize>().unwrap());
                            },
                            Rule::property_texname => {texname = Some(property.into_inner().next().unwrap().as_str().trim_matches(['\"', '\'']))},
                            Rule::property_antitexname => {antitexname = Some(property.into_inner().next().unwrap().as_str().trim_matches(['\"', '\'']))},
                            Rule::property_line => {
                                linestyle = Some(match property.as_str() {
                                    "dashed" => LineStyle::Dashed,
                                    "dotted" => LineStyle::Dotted,
                                    "straight" => LineStyle::Straight,
                                    "wavy" => LineStyle::Wavy,
                                    "curly" => LineStyle::Curly,
                                    "scurly" => LineStyle::Scurly,
                                    "swavy" => LineStyle::Swavy,
                                    "double" => LineStyle::Double,
                                    _ => return Err(ContentError(format!("Encountered unknown 'linestyle': {}", property.as_str())))
                                })
                            },
                            Rule::property_spin => {
                                twospin = Some(property.into_inner().next().unwrap().as_str().parse::<i8>().unwrap()-1);
                            },
                            Rule::property_color => {
                                color = Some(property.into_inner().next().unwrap().as_str().parse::<u8>().unwrap());
                            },
                            Rule::property_mass => (),
                            Rule::property_width => (),
                            _ => ()
                        }
                    }
                    if linestyle.is_none() {
                        if twospin.is_none() || color.is_none() {
                            return Err(ContentError(format!("Illegal particle definition in model {:#?} at position {:?}: \
                            either 'line' or 'spin' and 'color' is required", path, position)));
                        }
                        linestyle = match twospin.unwrap() {
                            0 => Some(LineStyle::Dashed),
                            1 => {
                                if texname != antitexname {
                                    Some(LineStyle::Straight)
                                } else if color == Some(1) {
                                    Some(LineStyle::Swavy)
                                } else {
                                    Some(LineStyle::Scurly)
                                }
                            },
                            2 => {
                                if color == Some(1) {
                                    Some(LineStyle::Wavy)
                                } else {
                                    Some(LineStyle::Curly)
                                }
                            }
                            4 => Some(LineStyle::Double),
                            -2 => Some(LineStyle::Dotted),
                            _ => {
                                warn!("Found twospin '{}' for particle {} in model {:#?}, \
                                for which 'linestle' is undefined. Defaulting to 'dashed'.", &twospin.unwrap(), &name.unwrap(), path);
                                Some(LineStyle::Dashed)
                            }
                        }
                    }
                    if let Some(spin) = twospin {
                        if spin == -2 || spin % 2 == 1 {
                            statistic = Fermi;
                        }
                    }
                    if let None = antiname {
                        antiname = Some(name.unwrap())
                    }
                    let particle = Particle::new(
                        name.unwrap(),
                        antiname.unwrap(),
                        pdg_code.unwrap(),
                        texname.unwrap(),
                        antitexname.unwrap(),
                        linestyle.unwrap(),
                        statistic
                    );
                    particles.insert(name.unwrap().into(), particle.clone());
                    particle_id_map.insert(particle_id, name.unwrap().into());
                    if name != antiname {
                        let anti = particle.clone().into_anti();
                        particles.insert(antiname.unwrap().into(), anti);
                    }
                }
                Rule::anti_particle => {
                    let mut properties = particle_rule.into_inner();
                    let anti_id = properties.next().unwrap().as_str().trim_matches(['\"', '\'']).to_string();
                    let part_id = properties.next().unwrap().as_str().trim_matches(['\"', '\'']).to_string();
                    if !particle_id_map.contains_key(&part_id) {
                        return Err(ContentError(format!("Model contains anti particle {} of particle {}, but {} does not exist", anti_id, part_id, part_id)));
                    }
                    particle_id_map.insert(anti_id, particles[particle_id_map.get(&part_id).unwrap()].anti_name.clone());
                },
                Rule::EOI => (),
                _ => unreachable!()
            }
        }
        return Ok((particles, particle_id_map));
    }

    fn parse_coupling_orders(path: &Path) -> Result<Vec<String>, ModelError> {
        let mut coupling_orders: Vec<String> = Vec::new();
        let coupling_orders_py_content = std::fs::read_to_string(path.join("coupling_orders.py"))?;
        let parsed_content = UFOParser::parse(Rule::coupling_orders_py, &coupling_orders_py_content)?.next().unwrap();
        for coupling_order_rule in parsed_content.into_inner() {
            match coupling_order_rule.as_rule() {
                Rule::coupling_order => {
                    let mut name = String::new();
                    for property in coupling_order_rule.into_inner() {
                        match property.as_rule() {
                            Rule::name => (),
                            Rule::property_name => {
                                name = property.into_inner().next().unwrap().as_str()
                                        .trim_matches(['\"', '\'']).to_string();
                            },
                            Rule::property_coupling_order => (),
                            Rule::property_hierarchy => (),
                            _ => unreachable!()
                        }
                    }
                    coupling_orders.push(name);
                },
                Rule::EOI => (),
                _ => unreachable!()
            }
        }
        return Ok(coupling_orders);
    }

    fn parse_couplings(path: &Path) -> Result<IndexMap<String, HashMap<String, usize>>, ModelError> {
        let mut couplings: IndexMap<String, HashMap<String, usize>> = IndexMap::new();
        let couplings_py_content = std::fs::read_to_string(path.join("couplings.py"))?;
        let parsed_content = UFOParser::parse(Rule::couplings_py, &couplings_py_content)?.next().unwrap();
        for couping_rule in parsed_content.into_inner() {
            match couping_rule.as_rule() {
                Rule::coupling => {
                    let mut py_name = String::new();
                    let mut order: HashMap<String, usize> = HashMap::new();
                    for property in couping_rule.into_inner() {
                        match property.as_rule() {
                            Rule::name => {
                                py_name = property.as_str().to_string();
                            }
                            Rule::property_name => (),
                            Rule::property_order => {
                                for entry in property.into_inner() {
                                    let mut inner_rules = entry.into_inner();
                                    order.insert(
                                        inner_rules.next().unwrap().as_str().trim_matches(['\"', '\'']).to_string().to_string(),
                                        inner_rules.next().unwrap().as_str().parse::<usize>().unwrap()
                                    );
                                }
                            },
                            Rule::property_value => (),
                            _ => unreachable!()
                        }
                    }
                    if !order.is_empty() {
                        couplings.insert(py_name, order);
                    } else { 
                        return Err(ContentError(format!("Property 'order' is required for Coupling '{}'", py_name)));
                    }
            }
                Rule::EOI => (),
                _ => unreachable!()
            }
        }
        return Ok(couplings);
    }

    fn parse_vertices(path: &Path, id_map: &HashMap<String, String>) -> Result<IndexMap<String, InteractionVertex>, ModelError> {
        let mut vertices: IndexMap<String, InteractionVertex> = IndexMap::new();
        let coupling_map = Self::parse_couplings(path)?;
        let vertices_py_content = std::fs::read_to_string(path.join("vertices.py"))?;
        let parsed_content = UFOParser::parse(Rule::vertices_py, &vertices_py_content)?.next().unwrap();
        for vertex_rule in parsed_content.into_inner() {
            match vertex_rule.as_rule() {
                Rule::vertex => {
                    let mut name = String::new();
                    let mut particles: Vec<String> = Vec::new();
                    let mut coupling_orders_vec = Vec::new();
                    for property in vertex_rule.into_inner() {
                        match property.as_rule() {
                            Rule::name => (),
                            Rule::property_name => {
                                name = property.into_inner().next().unwrap().as_str()
                                        .trim_matches(['\"', '\'']).to_string();
                            },
                            Rule::property_particles => {
                                for particle in property.into_inner() {
                                    particles.push(id_map.get(&particle.as_str().to_string()).unwrap().clone());
                                }
                            },
                            Rule::property_couplings => {
                                for coupling in property.into_inner() {
                                    let current_orders = coupling_map.get(coupling.as_str())
                                        .unwrap().clone();
                                    if !coupling_orders_vec.contains(&current_orders) {
                                        coupling_orders_vec.push(current_orders);
                                    }
                                }
                                
                            },
                            Rule::property_lorentz => (),
                            Rule::property_vertex_color => (),
                            _ => unreachable!()
                        }
                    }
                    if !particles.is_empty() {
                        let n_distinct_orders = coupling_orders_vec.len();
                        match n_distinct_orders {
                            0 => return Err(ContentError(format!("Property 'order' required for vertex {}", &name))),
                            1 => {
                                vertices.insert(name.clone(),
                                                InteractionVertex {
                                        name,
                                        particles,
                                        couplings_orders: coupling_orders_vec.pop().unwrap()
                                    }
                                );
                            }
                            n => {
                                warn!("Found ambiguous coupling order structure for vertex {}, \
                                splitting into {} vertices.", &name, n);
                                for i in 0..n {
                                    vertices.insert(format!("{}_{}", &name, i),
                                                    InteractionVertex {
                                            name: format!("{}_{}", &name, i),
                                            particles: particles.clone(),
                                            couplings_orders: coupling_orders_vec.pop().unwrap()
                                        }
                                    );
                                }
                                
                            }
                        }
                    } else {
                        return Err(ContentError(format!("Property 'particles' required for vertex {}", name)));
                    }
                }
                Rule::EOI => (),
                _ => unreachable!()
            }
        }
        return Ok(vertices);
    }

    pub(crate) fn parse_ufo_model(path: &Path) -> Result<Model, ModelError> {
        let (particles, particle_id_map) = Self::parse_particles(path)?;
        let coupling_orders = Self::parse_coupling_orders(path)?;
        let vertices = Self::parse_vertices(path, &particle_id_map)?;
        return Ok(Model {
            particles,
            vertices,
            coupling_orders
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use indexmap::IndexMap;
    use crate::model::{Model, ufo_parser::UFOParser, Particle, InteractionVertex, LineStyle, Statistic};

    #[test]
    fn ufo_parse_test() {
        let path = PathBuf::from("tests/QCD_UFO");
        let model = UFOParser::parse_ufo_model(&path).unwrap();
        println!("{:#?}", model);
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
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_2".to_string(), InteractionVertex {
                    name: "V_2".to_string(),
                    particles: vec!["G".to_string(); 4],
                    couplings_orders: HashMap::from([("QCD".to_string(), 2)])
                }), 
                ("V_3".to_string(), InteractionVertex {
                    name: "V_3".to_string(),
                    particles: vec!["u~".to_string(), "u".to_string(), "G".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_4".to_string(), InteractionVertex {
                    name: "V_4".to_string(),
                    particles: vec!["c~".to_string(), "c".to_string(), "G".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_5".to_string(), InteractionVertex {
                    name: "V_5".to_string(),
                    particles: vec!["t~".to_string(), "t".to_string(), "G".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                })
            ]),
            coupling_orders: vec!["QCD".to_string()],
        };
        println!("{:#?}", model_ref);
        assert_eq!(model, model_ref);
    }
}