use criterion::{Criterion, criterion_group, criterion_main};
use feyngraph::prelude::*;

fn diag_generator_2loop_bench(c: &mut Criterion) {
    let model = Model::default();
    let diag_gen = DiagramGenerator::new(
        vec![model.get_particle_index("u").unwrap().clone(); 2],
        vec![
            model.get_particle_index("u").unwrap().clone(),
            model.get_particle_index("u").unwrap().clone(),
            model.get_particle_index("g").unwrap().clone(),
        ],
        2,
        model,
        None,
    );
    c.bench_function("Diagram Generator 2-loop", |b| b.iter(|| diag_gen.generate()));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = diag_generator_2loop_bench
);
criterion_main!(benches);
