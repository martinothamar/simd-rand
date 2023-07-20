#![feature(stdsimd)]

use criterion::black_box;
use rand_core::SeedableRng;
use simd_prng::specific::avx512::*;
use std::arch::x86_64::*;

#[inline(never)]
fn do_xoshiro_mm512d(rng: &mut Xoshiro256PlusX8) -> __m512d {
    unsafe {
        let mut result = _mm512_set1_pd(0.0);
        rng.next_m512d(&mut result);
        result
    }
}

fn main() {
    let mut rng = Xoshiro256PlusX8::seed_from_u64(0);

    black_box(do_xoshiro_mm512d(&mut rng));
}
