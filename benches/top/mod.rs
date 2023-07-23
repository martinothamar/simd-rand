use std::{mem, arch::x86_64::*};

use criterion::{measurement::Measurement, Criterion, Throughput, black_box, BenchmarkId};
use rand::Rng;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use simd_rand::specific::avx512::{U64x8, SimdRand, Xoshiro256PlusX8};
use packed_simd_2::u64x8 as ps_u64x8;

#[inline(always)]
fn execute_rand<RNG: RngCore>(rng: &mut RNG, data: &mut U64x8, i: usize) {
    for _ in 0..i {
        for i in 0..8 {
            black_box(&mut *data)[i] = rng.next_u64();
        }
    }
}

#[inline(always)]
fn execute_rand_vectorized<RNG: RngCore>(rng: &mut RNG, data: &mut ps_u64x8, i: usize) {
    for _ in 0..i {
        *black_box(&mut *data) = rng.gen::<ps_u64x8>();
    }
}

#[inline(always)]
fn execute_vectorized<RNG: SimdRand>(rng: &mut RNG, data: &mut __m512i, i: usize) {
    for _ in 0..i {
        *black_box(&mut *data) = rng.next_m512i();
    }
}

pub fn add_top_benchmark<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("top");

    let iterations: Vec<_> = (0..8).map(|v| (v + 1) * ITERATIONS).collect();

    for iterations in iterations {
        group.throughput(Throughput::Bytes((iterations * mem::size_of::<__m512i>()) as u64));
        
        let name = BenchmarkId::new(format!("Rand/Xoshiro256+"), iterations);
        group.bench_with_input(name, &iterations, |b, i| {
            let mut rng = Xoshiro256Plus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data = Default::default();

            b.iter(|| execute_rand(&mut rng, black_box(&mut data), black_box(*i)))
        });

        let name = BenchmarkId::new(format!("RandVectorized/Xoshiro256+"), iterations);
        group.bench_with_input(name, &iterations, |b, i| {
            let mut rng = Xoshiro256Plus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data = Default::default();

            b.iter(|| execute_rand_vectorized(&mut rng, black_box(&mut data), black_box(*i)))
        });
        
        let name = BenchmarkId::new(format!("AVX512/Xoshiro256+"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512i = _mm512_setzero_si512();

            b.iter(|| execute_vectorized(&mut rng, black_box(&mut data), black_box(*i)))
        });
    }

    group.finish();
}
