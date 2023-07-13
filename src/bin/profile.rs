use std::{hint::black_box, time::Instant};

use rand_core::{SeedableRng, RngCore};
use shishua::Shishua;

fn main() {
    let mut rng: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

    let start = Instant::now();
    for _ in 0..4_000_000_000u64 {
        black_box( rng.next_u64());
    }
    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("Took {duration:?}");
}