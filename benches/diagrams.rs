use criterion::{Criterion, criterion_group, criterion_main};
use feyngraph::*;

fn diag_generator_2loop_bench(c: &mut Criterion) {
    let model = Model::default();
    let diag_gen = DiagramGenerator::new(&["u"; 2], &["u", "u", "g"], 2, model, None).unwrap();
    c.bench_function("Diagram Generator 2-loop", |b| b.iter(|| diag_gen.generate()));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = diag_generator_2loop_bench
);
criterion_main!(benches);
