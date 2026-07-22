#![allow(dead_code, non_snake_case)]
use feyngraph::{
    model::{Model, ModelBase, ParticleBase, VertexBase},
    topology::TopologyModel,
};
use regex::Regex;
use std::error::Error;
use std::io::{Error as IOError, ErrorKind};
use std::process::Command;

pub fn write_qgraf_model<M: ModelBase>(mut out: impl std::io::Write, model: &Model<M>) -> Result<(), Box<dyn Error>> {
    writeln!(out, "% Particles")?;
    for particle in model.particles() {
        if particle.id() < 0 {
            continue;
        }
        if particle.is_self_anti() {
            writeln!(
                out,
                "[ part{}, part{}, {}]",
                particle.id(),
                particle.id(),
                if particle.is_fermi() { "-" } else { "+" }
            )?;
        } else {
            writeln!(
                out,
                "[ part{}, anti{}, {}]",
                particle.id(),
                particle.id(),
                if particle.is_fermi() { "-" } else { "+" }
            )?;
        }
    }
    writeln!(out, "% Vertices")?;
    let n = model.couplings().len() - 1;
    for vertex in model.vertices() {
        write!(out, "[ ")?;
        for (i, particle) in vertex.particles().iter().enumerate() {
            let p = model.particle_by_name(particle).unwrap();
            if p.id() > 0 {
                write!(out, "part{}", p.id())?;
            } else {
                write!(out, "anti{}", p.id().abs())?;
            }
            if i != vertex.degree() - 1 {
                write!(out, ", ")?;
            }
        }
        write!(out, "; ")?;
        let coupling_orders = vertex.coupling_orders();
        for (i, coupling) in model.couplings().enumerate() {
            if let Some(order) = coupling_orders.get(coupling.as_ref()) {
                write!(out, "{}='{}'", coupling.as_ref(), order)?;
            } else {
                write!(out, "{}='0'", coupling.as_ref())?;
            }
            if i != n {
                write!(out, ", ")?;
            }
        }
        writeln!(out, "]")?;
    }
    Ok(())
}

pub fn write_qgraf_topo_model(mut out: impl std::io::Write, model: &TopologyModel) -> Result<(), Box<dyn Error>> {
    writeln!(out, "[phi, phi, +; ID='1']")?;
    for degree in model.degrees_iter() {
        writeln!(out, "[ {}{} ]", "phi, ".repeat(degree - 1), "phi")?;
    }
    Ok(())
}

pub fn write_qgraf_config(
    mut out: impl std::io::Write,
    model_path: &str,
    in_particles: &Vec<String>,
    out_particles: &Vec<String>,
    n_loops: usize,
    options: &[&str],
) -> Result<(), Box<dyn Error>> {
    writeln!(
        out,
        "\
config = nolist ;
style = 'tests/resources/empty.sty' ;
output = 'out' ;
model = '{}' ;
in = {};
out = {};
loops = {};
options = {};
    ",
        model_path,
        in_particles.join(", "),
        out_particles.join(", "),
        n_loops,
        options.join(", ")
    )?;
    Ok(())
}

pub fn run_qgraf(config_path: &str) -> Result<usize, Box<dyn Error>> {
    let output = Command::new("qgraf").arg(config_path).output()?;
    let re = Regex::new(r"total\s*=\s*(\d+)\s+connected diagrams").unwrap();
    let res = re.captures(&std::str::from_utf8(&output.stdout)?);
    if res.is_none() {
        println!("{}", std::str::from_utf8(&output.stdout)?);
        return Err(Box::new(IOError::new(
            ErrorKind::Other,
            std::str::from_utf8(&output.stdout)?,
        )));
    }
    Ok(res.unwrap()[1].parse()?)
}
