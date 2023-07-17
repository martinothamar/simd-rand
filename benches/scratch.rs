#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::arch::x86_64::{__m256d, _mm256_set1_pd};
use std::mem;

use criterion::measurement::Measurement;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use criterion_perf_events::Perf;
use perfcnt::linux::HardwareEventType as Hardware;
use perfcnt::linux::PerfCounterBuilderLinux as Builder;
use rand::Rng;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use simd_prng::specific::avx2::*;

const ITERATIONS: usize = 16;


fn do_next_m256d(rng: &mut Xoshiro256PlusPlusX4, data: &mut __m256d) {
    for _ in 0..ITERATIONS {
        rng.next_m256d(black_box(data));
    }
}
fn do_next_m256d_pure_avx(rng: &mut Xoshiro256PlusPlusX4, data: &mut __m256d) {
    for _ in 0..ITERATIONS {
        rng.next_m256d_pure_avx(black_box(data));
    }
}

fn bench<M: Measurement, const T: u8>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("scratch");

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<U64x4>()) as u64));

    let suffix = match T {
        Type::TIME => "Time",
        Type::INST => "Instructions",
        Type::CYCLES => "Cycles",
        _ => unreachable!(),
    };

    let next_m256d_name = format!("next_m256d __m256d - {suffix}");
    let next_m256d_from_f64x4_name = format!("next_m256d_pure_avx __m256d - {suffix}");


    group.bench_function(next_m256d_name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data = _mm256_set1_pd(0.0);
    
            b.iter(|| do_next_m256d(&mut rng, black_box(&mut data)))
        }
    });

    group.bench_function(next_m256d_from_f64x4_name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data = _mm256_set1_pd(0.0);
    
            b.iter(|| do_next_m256d_pure_avx(&mut rng, black_box(&mut data)))
        }
    });

    group.finish();
}

#[non_exhaustive]
struct Type;

impl Type {
    pub const TIME: u8 = 1;
    pub const INST: u8 = 2;
    pub const CYCLES: u8 = 3;
}

criterion_group!(
    name = time;
    config = Criterion::default();
    targets = bench::<_, { Type::TIME }>
);
// criterion_group!(
//     name = instructions;
//     config = Criterion::default().with_measurement(Perf::new(Builder::from_hardware_event(Hardware::Instructions)));
//     targets = bench::<_, { Type::INST }>
// );
// criterion_group!(
//     name = cycles;
//     config = Criterion::default().with_measurement(Perf::new(Builder::from_hardware_event(Hardware::RefCPUCycles)));
//     targets = bench::<_, { Type::CYCLES }>
// );
criterion_main!(time);
