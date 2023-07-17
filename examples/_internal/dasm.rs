use criterion::black_box;
use rand_core::SeedableRng;
use simd_prng::specific::avx2::*;
use std::arch::x86_64::*;

#[inline(never)]
fn do_xoshiro_mm256d(rng: &mut Xoshiro256PlusPlusX4) -> __m256d {
    unsafe {
        let mut result = _mm256_set1_pd(0.0);
        rng.next_m256d(&mut result);
        result
    }
}

#[inline(never)]
fn do_xoshiro_mm256d_pure_avx(rng: &mut Xoshiro256PlusPlusX4) -> __m256d {
    unsafe {
        let mut result = _mm256_set1_pd(0.0);
        rng.next_m256d_pure_avx(&mut result);
        result
    }
}

fn main() {
    let mut rng = Xoshiro256PlusPlusX4::seed_from_u64(0);

    black_box(do_xoshiro_mm256d(&mut rng));
    black_box(do_xoshiro_mm256d_pure_avx(&mut rng));
}
