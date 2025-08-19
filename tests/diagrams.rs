#![allow(dead_code, non_snake_case)]
use feyngraph::{
    diagram::DiagramGenerator,
    model::{Model, TopologyModel},
    topology::TopologyGenerator,
};
use itertools::Itertools;
use paste::paste;
use std::path::PathBuf;
use tempfile::NamedTempFile;
mod common;

macro_rules! test_diagrams {
    (
        $ufo:ident,
        $($particle_in:ident)+ => $($particle_out:ident)+,
        $n_loops:literal
    ) => {
        paste!{
            #[test]
            fn [< test_diag _ $ufo _ $($particle_in)+ _ $($particle_out)+ _loops_ $n_loops >]() {
                let model = Model::from_ufo(
                    &PathBuf::from(format!("tests/resources/{}", stringify!($ufo)))
                ).unwrap();
                let in_particles = [$(stringify!($particle_in)),+];
                let out_particles = [$(stringify!($particle_out)),+];
                let n_diags = DiagramGenerator::new(
                    &in_particles,
                    &out_particles,
                    $n_loops,
                    model.clone(),
                    None,
                ).unwrap().count();
                let qgraf_model = NamedTempFile::with_prefix("qgraf_model_").unwrap();
                common::write_qgraf_model(
                    &qgraf_model,
                    &model
                ).unwrap();
                let in_particles = vec![$(model.get_particle_by_name(stringify!($particle_in)).unwrap()),+];
                let out_particles = vec![$(model.get_particle_by_name(stringify!($particle_out)).unwrap()),+];
                let qgraf_config = NamedTempFile::with_prefix("qgraf_config_").unwrap();
                common::write_qgraf_config(
                    &qgraf_config,
                    &qgraf_model.path().to_str().unwrap(),
                    &in_particles.iter().map(
                        |p| if p.pdg() > 0 {format!("part{}", p.pdg())} else {format!("anti{}", p.pdg().abs())}
                    ).collect_vec(),
                    &out_particles.iter().map(
                        |p| if p.pdg() > 0 {format!("part{}", p.pdg())} else {format!("anti{}", p.pdg().abs())}
                    ).collect_vec(),
                    $n_loops,
                    &vec![]
                ).unwrap();
                let n_qgraf = common::run_qgraf(qgraf_config.path().to_str().unwrap());
                if let Ok(n_qgraf) = n_qgraf {
                    assert_eq!(n_qgraf, n_diags);
                } else if let Err(e) = n_qgraf {
                    println!("{}", std::fs::read_to_string(qgraf_model.path().to_str().unwrap()).unwrap());
                    println!("{}", std::fs::read_to_string(qgraf_config.path().to_str().unwrap()).unwrap());
                    println!("{:#?}", e);
                    panic!("QGRAF terminated due to an error");
                }
            }
        }
    };
    (
        $ufo:ident,
        $name:ident,
        $($particle_in:literal)+ => $($particle_out:literal)+,
        $n_loops:literal
    ) => {
        paste!{
            #[test]
            fn [< test_diag _ $ufo _ $name _loops_ $n_loops >]() {
                let model = Model::from_ufo(
                    &PathBuf::from(format!("tests/resources/{}", stringify!($ufo)))
                ).unwrap();
                let in_particles = vec![$($particle_in),+];
                let out_particles = vec![$($particle_out),+];
                let n_diags = DiagramGenerator::new(
                    &in_particles,
                    &out_particles,
                    $n_loops,
                    model.clone(),
                    None,
                ).unwrap().count();
                let qgraf_model = NamedTempFile::with_prefix("qgraf_model_").unwrap();
                common::write_qgraf_model(
                    &qgraf_model,
                    &model
                ).unwrap();
                let in_particles = vec![$(model.get_particle_by_name($particle_in).unwrap()),+];
                let out_particles = vec![$(model.get_particle_by_name($particle_out).unwrap()),+];
                let qgraf_config = NamedTempFile::with_prefix("qgraf_config_").unwrap();
                common::write_qgraf_config(
                    &qgraf_config,
                    &qgraf_model.path().to_str().unwrap(),
                    &in_particles.iter().map(
                        |p| if p.pdg() > 0 {format!("part{}", p.pdg())} else {format!("anti{}", p.pdg().abs())}
                    ).collect_vec(),
                    &out_particles.iter().map(
                        |p| if p.pdg() > 0 {format!("part{}", p.pdg())} else {format!("anti{}", p.pdg().abs())}
                    ).collect_vec(),
                    $n_loops,
                    &vec![]
                ).unwrap();
                let n_qgraf = common::run_qgraf(qgraf_config.path().to_str().unwrap());
                if let Ok(n_qgraf) = n_qgraf {
                    assert_eq!(n_qgraf, n_diags);
                } else if let Err(e) = n_qgraf {
                    println!("{}", std::fs::read_to_string(qgraf_model.path().to_str().unwrap()).unwrap());
                    println!("{}", std::fs::read_to_string(qgraf_config.path().to_str().unwrap()).unwrap());
                    println!("{:#?}", e);
                    panic!("QGRAF terminated due to an error");
                }
            }
        }
    };
}

test_diagrams!(Standard_Model_UFO, g g => g g, 0);
test_diagrams!(Standard_Model_UFO, g g => g g, 1);
test_diagrams!(Standard_Model_UFO, g g => g g, 2);

test_diagrams!(Standard_Model_UFO, u u => u u, 0);
test_diagrams!(Standard_Model_UFO, u u => u u, 1);
test_diagrams!(Standard_Model_UFO, u u => u u, 2);

test_diagrams!(Standard_Model_UFO, uubar_uubar, "u" "u~" => "u" "u~", 0);
test_diagrams!(Standard_Model_UFO, uubar_uubar, "u" "u~" => "u" "u~", 1);
test_diagrams!(Standard_Model_UFO, uubar_uubar, "u" "u~" => "u" "u~", 2);

test_diagrams!(Standard_Model_UFO, gg_uubar, "g" "g" => "u" "u~", 0);
test_diagrams!(Standard_Model_UFO, gg_uubar, "g" "g" => "u" "u~", 1);
test_diagrams!(Standard_Model_UFO, gg_uubar, "g" "g" => "u" "u~", 2);

test_diagrams!(Standard_Model_UFO, u u => u u g, 0);
test_diagrams!(Standard_Model_UFO, u u => u u g, 1);
test_diagrams!(Standard_Model_UFO, u u => u u g, 2);

test_diagrams!(QCD_UFO, G => G, 3);
test_diagrams!(QCD_UFO, G => G, 4);

test_diagrams!(QCD_UFO, u => u, 3);
test_diagrams!(QCD_UFO, u => u, 4);

#[test]
fn diags_QCD_4F_UFO_uubar_uubar_loops_0() {
    let model = Model::from_ufo(&PathBuf::from("tests/resources/QCD_4F_UFO")).unwrap();
    let in_particles = ["u", "u~"];
    let out_particles = ["u", "u~"];
    let topos = TopologyGenerator::new(4, 0, TopologyModel::from(&model), None).generate();
    let generator = DiagramGenerator::new(&in_particles, &out_particles, 0, model.clone(), None).unwrap();
    let diags = generator.assign_topology(&topos[0]).unwrap();
    assert_eq!(2, diags.len());
}

#[test]
fn diags_SMEFT_epem_epem_loops_2() {
    let model = Model::from_ufo(&PathBuf::from("tests/resources/SMEFT")).unwrap();
    let in_particles = ["e+", "e-"];
    let out_particles = ["e+", "e-"];
    let topos = TopologyGenerator::new(4, 2, TopologyModel::from(&model), None).generate();
    let generator = DiagramGenerator::new(&in_particles, &out_particles, 2, model, None).unwrap();
    let diags = generator.assign_topology(&topos[34]).unwrap();
    assert!(diags.len() > 0);
}

#[test]
fn diags_SM_uu_uu__loops_1() {
    let model = Model::default();
    let diag_gen = DiagramGenerator::new(&["u"; 2], &["u", "u", "g"], 1, model, None).unwrap();
    diag_gen.generate();
}
