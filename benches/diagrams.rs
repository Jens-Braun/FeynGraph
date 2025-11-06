use criterion::{Criterion, criterion_group, criterion_main};
use feyngraph::*;

fn diag_generator_2loop_bench(c: &mut Criterion) {
    let model = Model::default();
    let diag_gen = DiagramGenerator::new(&["u"; 2], &["u", "u", "g"], 2, model, None).unwrap();
    c.bench_function("Diagram Generator 2-loop", |b| b.iter(|| diag_gen.generate()));
}

fn diag_generator_6g_2loop_bench(c: &mut Criterion) {
    let model = Model::default();
    let mut sel = DiagramSelector::new();
    sel.select_on_shell();
    sel.select_self_loops(0);
    let diag_gen = DiagramGenerator::new(&["g"; 2], &["g"; 4], 2, model, Some(sel)).unwrap();
    c.bench_function("Diagram Generator 6g 2-loop", |b| b.iter(|| diag_gen.count()));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = diag_generator_2loop_bench, diag_generator_6g_2loop_bench
);
criterion_main!(benches);
