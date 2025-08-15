#![allow(dead_code, non_snake_case)]
use feyngraph::{model::TopologyModel, topology::TopologyGenerator};
use paste::paste;
use tempfile::NamedTempFile;

mod common;

macro_rules! test_topo {
    (
        [$($degree:literal),*],
        $n_legs:literal,
        $n_loops:literal
    ) => {
        paste!{
            #[test]
            fn [< test_topo $(_$degree)* _legs_ $n_legs _loops_ $n_loops >]() {
                let model = TopologyModel::from(vec![$($degree,)*]);
                let n_topos = TopologyGenerator::new(
                    $n_legs,
                    $n_loops,
                    model.clone(),
                    None,
                ).count_topologies();
                let qgraf_model = NamedTempFile::with_prefix("qgraf_model_").unwrap();
                common::write_qgraf_topo_model(
                    &qgraf_model,
                    &model
                ).unwrap();
                let qgraf_config = NamedTempFile::with_prefix("qgraf_config_").unwrap();
                common::write_qgraf_config(
                    &qgraf_config,
                    &qgraf_model.path().to_str().unwrap(),
                    &vec![String::from("phi"); $n_legs],
                    &vec![],
                    $n_loops,
                    &vec![]
                ).unwrap();
                let n_qgraf = common::run_qgraf(qgraf_config.path().to_str().unwrap());
                if let Ok(n_qgraf) = n_qgraf {
                    assert_eq!(n_qgraf, n_topos);
                } else {
                    println!("{}", std::fs::read_to_string(qgraf_model.path()).unwrap());
                    println!("{}", std::fs::read_to_string(qgraf_config.path()).unwrap());
                    println!("{:?}", n_qgraf);
                    panic!("QGRAF terminated due to an error")
                }
            }
        }
    };
}

test_topo!([3, 4], 2, 1);
test_topo!([3, 4], 2, 2);
test_topo!([3, 4], 2, 3);
test_topo!([3, 4], 2, 4);
test_topo!([3, 4], 2, 5);

test_topo!([3, 4], 4, 0);
test_topo!([3, 4], 4, 1);
test_topo!([3, 4], 4, 2);
test_topo!([3, 4], 4, 3);

test_topo!([3, 4], 6, 0);
test_topo!([3, 4], 6, 1);
test_topo!([3, 4], 6, 2);

test_topo!([3, 4], 8, 0);
test_topo!([3, 4], 8, 1);

test_topo!([3, 4, 5], 5, 0);
test_topo!([3, 4, 5], 5, 1);
test_topo!([3, 4, 5], 5, 2);
test_topo!([3, 4, 5], 5, 3);

test_topo!([3, 4, 5, 6], 4, 0);
test_topo!([3, 4, 5, 6], 4, 1);
test_topo!([3, 4, 5, 6], 4, 2);

#[test]
fn test_topo_3_4_legs_0_loops_2() {
    let model = TopologyModel::from(vec![3, 4]);
    let n_topos = TopologyGenerator::new(0, 2, model, None).count_topologies();
    assert_eq!(n_topos, 3);
}

#[test]
fn test_topo_3_4_legs_0_loops_3() {
    let model = TopologyModel::from(vec![3, 4]);
    let n_topos = TopologyGenerator::new(0, 3, model, None).count_topologies();
    assert_eq!(n_topos, 12);
}
