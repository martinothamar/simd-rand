use std::{mem, arch::x86_64::*};

use criterion::{measurement::Measurement, Criterion, Throughput, black_box};
use rand_core::SeedableRng;
use simd_rand::specific::avx512::*;

pub fn add_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, suffix: &str) {
    let group_prefix = "AVX512";
    add_m512i_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
    add_m512d_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
}

fn add_m512i_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, group_prefix: &str, suffix: &str) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m512i"));

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<U64x8>()) as u64));

    #[inline(always)]
    fn execute<RNG: SimdPrng, const ITERATIONS: usize>(rng: &mut RNG, data: &mut __m512i) {
        for _ in 0..ITERATIONS {
            rng.next_m512i(black_box(data));
        }
    }

    let name = format!("Xoshiro256++ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusPlusX8 = Xoshiro256PlusPlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512i = _mm512_setzero_si512();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });

    let name = format!("Xoshiro256+ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusX8 = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512i = _mm512_setzero_si512();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });
}

fn add_m512d_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, group_prefix: &str, suffix: &str) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m512d"));

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<F64x8>()) as u64));

    #[inline(always)]
    fn execute<RNG: SimdPrng, const ITERATIONS: usize>(rng: &mut RNG, data: &mut __m512d) {
        for _ in 0..ITERATIONS {
            rng.next_m512d(black_box(data));
        }
    }

    let name = format!("Xoshiro256++ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusPlusX8 = Xoshiro256PlusPlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512d = _mm512_setzero_pd();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });

    let name = format!("Xoshiro256+ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusX8 = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512d = _mm512_setzero_pd();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });
}
