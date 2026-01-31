use rand_core::{RngCore, SeedableRng};
use simd_rand::portable::*;
use std::{hint::black_box, time::Instant};

fn main() {
    let mut seed: Xoshiro256PlusPlusX8Seed = Default::default();
    rand::rng().fill_bytes(&mut *seed);
    let mut rng = Xoshiro256PlusPlusX8::from_seed(seed);

    for _ in 0..4 {
        let start = Instant::now();
        for _ in 0..4_000_000_000u64 {
            black_box(rng.next_u64x8());
        }
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("Took {duration:?}");
    }
}
