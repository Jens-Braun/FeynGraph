use std::collections::HashMap;
use std::path::PathBuf;
use pest::Parser;
use pest_derive::Parser;
use crate::model::{LineStyle, Model, ModelError, Particle, Vertex};
use log::warn;
use crate::model::ModelError::{ContentError};

#[derive(Parser)]
#[grammar = "model/ufo_model.pest"]
pub(crate) struct UFOParser;

impl UFOParser {
    fn parse_particles(path: &PathBuf) -> Result<HashMap<String, Particle>, ModelError> {
        let mut particles: HashMap<String, Particle> = HashMap::new();
        let particle_py_content = std::fs::read_to_string(path.join("particles.py"))?;
        let parsed_content = UFOParser::parse(Rule::particles_py, &particle_py_content)?.next().unwrap();
        for particle_rule in parsed_content.into_inner() {
            match particle_rule.as_rule() {
                Rule::particle => {
                    let mut pdg_code = None;
                    let mut name = None;
                    let mut texname = None;
                    let mut antitexname = None;
                    let mut linestyle = None;
                    let mut spin = None;
                    let mut color = None;
                    let position = particle_rule.line_col();
                    for property in particle_rule.into_inner() {
                        match property.as_rule() {
                            Rule::property_pdg_code => {
                                pdg_code = Some(property.into_inner().next().unwrap().as_str().parse::<isize>().unwrap());
                            },
                            Rule::property_name => {name = Some(property.into_inner().next().unwrap().as_str().trim_matches(['\"', '\'']))},
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
                                spin = Some(property.into_inner().next().unwrap().as_str().parse::<isize>().unwrap());
                            },
                            Rule::property_color => {
                                color = Some(property.into_inner().next().unwrap().as_str().parse::<usize>().unwrap());
                            },
                            _ => ()
                        }
                    }
                    if linestyle == None {
                        if spin == None || color == None {
                            return Err(ContentError(format!("Illegal particle definition in model {:#?} at position {:?}: \
                            either 'line' or 'spin' and 'color' is required", path, position)));
                        }
                        linestyle = match spin.unwrap() {
                            1 => Some(LineStyle::Dashed),
                            2 => {
                                if texname != antitexname {
                                    Some(LineStyle::Straight)
                                } else if color == Some(1) {
                                    Some(LineStyle::Swavy)
                                } else {
                                    Some(LineStyle::Scurly)
                                }
                            },
                            3 => {
                                if color == Some(1) {
                                    Some(LineStyle::Wavy)
                                } else {
                                    Some(LineStyle::Curly)
                                }
                            }
                            5 => Some(LineStyle::Double),
                            -1 => Some(LineStyle::Dotted),
                            _ => {
                                warn!("Found spin '{}' for particle {} in model {:#?}, \
                                for which 'linestle' is undefined. Defaulting to 'dashed'.", &spin.unwrap(), &name.unwrap(), path);
                                Some(LineStyle::Dashed)
                            }
                        }
                    }
                    particles.insert(name.unwrap().into(), Particle::new(
                        name.unwrap(),
                        pdg_code.unwrap(),
                        texname.unwrap(),
                        antitexname.unwrap(),
                        linestyle.unwrap(),
                    ));
                }
                Rule::anti_particle => (),
                Rule::EOI => (),
                _ => unreachable!()
            }
        }
        return Ok(particles);
    }

    fn parse_coupling_orders(path: &PathBuf) -> Result<Vec<String>, ModelError> {
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

    fn parse_couplings(path: &PathBuf) -> Result<HashMap<String, HashMap<String, usize>>, ModelError> {
        let mut couplings: HashMap<String, HashMap<String, usize>> = HashMap::new();
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
                    if order.len() != 0 {
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

    fn parse_vertices(path: &PathBuf) -> Result<HashMap<String, Vertex>, ModelError> {
        let mut vertices: HashMap<String, Vertex> = HashMap::new();
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
                                    particles.push(particle.as_str().to_string());
                                }
                            },
                            Rule::property_couplings => {
                                for coupling in property.into_inner() {
                                    let current_orders = coupling_map.get(&coupling.as_str().to_string())
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
                    if particles.len() != 0 {
                        let n_distinct_orders = coupling_orders_vec.len();
                        match n_distinct_orders {
                            0 => return Err(ContentError(format!("Property 'order' required for vertex {}", &name))),
                            1 => {
                                vertices.insert(name.clone(),
                                    Vertex {
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
                                        Vertex {
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

    pub(crate) fn parse_ufo_model(path: &PathBuf) -> Result<Model, ModelError> {
        let particles = Self::parse_particles(path)?;
        let coupling_orders = Self::parse_coupling_orders(path)?;
        let vertices = Self::parse_vertices(path)?;
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
    use crate::model::{Model, ufoparser::UFOParser, Particle, Vertex, LineStyle};

    #[test]
    fn ufo_parse_test() {
        let path = PathBuf::from("tests/QCD_UFO");
        let model = UFOParser::parse_ufo_model(&path).unwrap();
        println!("{:#?}", model);
        let model_ref = Model {
            particles: HashMap::from([
                (String::from("u"), Particle::new(
                    "u", 9000001, "u", "u~", LineStyle::Straight
                )),
                (String::from("c"), Particle::new(
                    "c", 9000002, "c", "c~", LineStyle::Straight
                )),
                (String::from("t"), Particle::new(
                    "t", 9000003, "t", "t~", LineStyle::Straight
                )),
                (String::from("G"), Particle::new(
                    "G", 9000004, "G", "G", LineStyle::Curly
                ))

            ]),
            vertices: HashMap::from([
                ("V_1".to_string(), Vertex {
                    name: "V_1".to_string(),
                    particles: vec!["G".to_string(); 3],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_2".to_string(), Vertex {
                    name: "V_2".to_string(),
                    particles: vec!["G".to_string(); 4],
                    couplings_orders: HashMap::from([("QCD".to_string(), 2)])
                }), 
                ("V_3".to_string(), Vertex {
                    name: "V_3".to_string(),
                    particles: vec!["u__tilde__".to_string(), "u".to_string(), "G".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_4".to_string(), Vertex {
                    name: "V_4".to_string(),
                    particles: vec!["c__tilde__".to_string(), "c".to_string(), "G".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                }),
                ("V_5".to_string(), Vertex {
                    name: "V_5".to_string(),
                    particles: vec!["t__tilde__".to_string(), "t".to_string(), "G".to_string()],
                    couplings_orders: HashMap::from([("QCD".to_string(), 1)])
                })
            ]),
            coupling_orders: vec!["QCD".to_string()],
        };
        println!("{:#?}", model_ref);
        assert_eq!(model, model_ref);
    }
}