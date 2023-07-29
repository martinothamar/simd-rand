use std::{mem, arch::x86_64::*, simd::u64x8};

use criterion::{measurement::Measurement, Criterion, Throughput, black_box, BenchmarkId};
use rand::Rng;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use simd_rand::portable::*;
use simd_rand::specific;
use packed_simd_2::u64x8 as ps_u64x8;

#[inline(never)]
fn execute_rand<RNG: RngCore>(rng: &mut RNG, data: &mut u64x8) {
    for i in 0..8 {
        data[i] = rng.next_u64();
    }
}

#[inline(never)]
fn execute_rand_vectorized<RNG: RngCore>(rng: &mut RNG, data: &mut ps_u64x8) {
    *data = rng.gen::<ps_u64x8>();
}

#[inline(never)]
fn execute_vectorized_portable<RNG: SimdRandX8>(rng: &mut RNG, data: &mut u64x8) {
    *data = rng.next_u64x8();
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512dq"))]
#[inline(never)]
fn execute_vectorized_specific<RNG: specific::avx512::SimdRand>(rng: &mut RNG, data: &mut __m512i) {
    *data = rng.next_m512i();
}

pub fn add_top_benchmark<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Top");

    group.throughput(Throughput::Bytes(mem::size_of::<u64x8>() as u64));
        
    let name = BenchmarkId::new(format!("Rand/Xoshiro256+"), 1);

    group.bench_with_input(name, &1, |b, i| {
        let mut rng = Xoshiro256Plus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data = Default::default();

        b.iter(|| execute_rand(&mut rng, black_box(&mut data)))
    });

    let name = BenchmarkId::new(format!("RandVectorized/Xoshiro256+"), 1);
    group.bench_with_input(name, &1, |b, i| {
        let mut rng = Xoshiro256Plus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data = Default::default();

        b.iter(|| execute_rand_vectorized(&mut rng, black_box(&mut data)))
    });
    
    let name = BenchmarkId::new(format!("Portable/Xoshiro256+X8"), 1);
    group.bench_with_input(name, &1, |b, i| {
        let mut rng = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data = Default::default();

        b.iter(|| execute_vectorized_portable(&mut rng, black_box(&mut data)))
    });
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512dq"))]
    let name = BenchmarkId::new(format!("Specific/Xoshiro256+X8"), iterations);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512dq"))]
    group.bench_with_input(name, &iterations, |b, i| unsafe {
        let mut rng = specific::avx512::Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: __m512i = _mm512_setzero_si512();

        b.iter(|| execute_vectorized_specific(&mut rng, black_box(&mut data)))
    });

    group.finish();
}
