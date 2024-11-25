use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use pest::Parser;
use pest_derive::Parser;
use crate::model::{Coupling, LineStyle, Model, ModelError, Particle, Vertex};
use log::warn;
use crate::model::ModelError::{ContentError};

#[derive(Parser)]
#[grammar = "model/ufo_model.pest"]
struct UFOParser;

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
                    println!("{:?}", name);
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

    fn parse_coupling_orders(path: &PathBuf) -> Result<Vec<String>, Box<dyn Error>> {
        todo!();
    }

    fn parse_couplings(path: &PathBuf) -> Result<Vec<Coupling>, Box<dyn Error>> {
        todo!();
    }

    fn parse_vertices(path: &PathBuf) -> Result<Vec<Vertex>, Box<dyn Error>> {
        todo!();
    }

    fn parse_ufo_model(path: &PathBuf) -> Result<Model, Box<dyn Error>> {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use crate::model::{Model, ufoparser::UFOParser, Particle, LineStyle};

    #[test]
    fn ufo_parse_test() {
        let path = PathBuf::from("tests/QCD_UFO");
        let model = UFOParser::parse_ufo_model(&path).unwrap();
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
            vertices: Vec::new()
        };
        assert_eq!(model, model_ref);
    }
}