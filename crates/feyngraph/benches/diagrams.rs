use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use feyngraph::{
    model::ufo::{Reader, UFOModel},
    prelude::*,
};
use rayon::ThreadPoolBuilder;
use std::path::PathBuf;

fn diag_generator_1loop_bench(c: &mut Criterion) {
    let model = Model::from_base(
        UFOModel::read(
            &PathBuf::from("../../../tests/models/Standard_Model_UFO"),
            Reader::CPython,
        )
        .unwrap(),
    );
    let diag_gen = DiagramGenerator::new(&["u", "u~"], &["u", "u~"], 1, model, None).unwrap();

    let mut group = c.benchmark_group("Diagram Generator 1-loop");
    group.bench_function("Serial", |b| b.iter(|| diag_gen.generate()));
    for n_jobs in 1..=12 {
        group.bench_with_input(BenchmarkId::from_parameter(n_jobs), &n_jobs, |b, &n_jobs| {
            let pool = ThreadPoolBuilder::new().num_threads(n_jobs).build().unwrap();
            b.iter(|| pool.install(|| diag_gen.par_generate()));
        });
    }
    group.finish();
}

fn diag_generator_6g_1loop_bench(c: &mut Criterion) {
    let model = Model::from_base(
        UFOModel::read(
            &PathBuf::from("../../../tests/models/Standard_Model_UFO"),
            Reader::CPython,
        )
        .unwrap(),
    );
    let diag_gen = DiagramGenerator::new(&["g"; 2], &["g"; 2], 1, model, None).unwrap();
    c.bench_function("Diagram Generator 6g 1-loop", |b| b.iter(|| diag_gen.count()));
}

fn diag_generator_5g_2loop_bench(c: &mut Criterion) {
    let model = Model::from_base(
        UFOModel::read(
            &PathBuf::from("../../../tests/models/Standard_Model_UFO"),
            Reader::CPython,
        )
        .unwrap(),
    );
    let mut sel = DiagramSelector::new();
    sel.select_on_shell();
    sel.select_self_loops(0);
    let diag_gen = DiagramGenerator::new(&["g"; 2], &["g"; 3], 2, model, Some(sel)).unwrap();
    c.bench_function("Diagram Generator 5g 2-loop", |b| b.iter(|| diag_gen.count()));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = diag_generator_1loop_bench, diag_generator_6g_1loop_bench, diag_generator_5g_2loop_bench
);
criterion_main!(benches);
