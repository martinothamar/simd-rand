use std::mem;

use criterion::{criterion_group, criterion_main, Criterion, black_box, Throughput};
use rand::rngs::SmallRng;
use rand_core::{RngCore, SeedableRng};
use shishua::Shishua;

const ITERATIONS: usize = 64;

fn work<T: RngCore>(rng: &mut T) {
    for _ in 0..64 {
        black_box(rng.next_u64());
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RNG");

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<u64>()) as u64));

    group.bench_function("Shishua u64", |b| {
        let mut rng: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| work(&mut rng))
    });
    group.bench_function("SmallRng u64", |b| {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| work(&mut rng))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
