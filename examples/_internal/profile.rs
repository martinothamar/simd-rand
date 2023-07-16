use std::{hint::black_box, time::Instant};

use rand_core::{RngCore, SeedableRng};
use simd_prng::specific::avx2::*;

fn main() {
    let mut rng: Shishua<{ 1024 * 32 }> = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

    for _ in 0..4 {
        let start = Instant::now();
        for _ in 0..4_000_000_000u64 {
            black_box(rng.next_u64());
        }
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("Took {duration:?}");
    }
}
