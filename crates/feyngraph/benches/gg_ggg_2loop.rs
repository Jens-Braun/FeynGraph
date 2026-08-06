use criterion::{Criterion, criterion_group, criterion_main};
use feyngraph::prelude::*;

fn diag_gen_gg_ggg_2loop(c: &mut Criterion) {
    let model = Model::sm();
    let mut s = DiagramSelector::new();
    s.select_self_loops(2);
    let diag_gen = DiagramGenerator::new(&["g"; 2], &["g"; 3], 2, model, Some(s)).unwrap();
    c.bench_function("gg -> ggg @ 2-loops", |b| b.iter(|| diag_gen.count()));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = diag_gen_gg_ggg_2loop
);
criterion_main!(benches);
