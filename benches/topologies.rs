use criterion::{Criterion, criterion_group, criterion_main};
use feyngraph::{model::TopologyModel, topology::TopologyGenerator};

fn topo_generator_3loop_bench(c: &mut Criterion) {
    let topo_gen = TopologyGenerator::new(4, 3, TopologyModel::from(vec![3, 4]), None);
    c.bench_function("TopologyGenerator 3-loop", |b| b.iter(|| topo_gen.generate()));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = topo_generator_3loop_bench
);
criterion_main!(benches);
