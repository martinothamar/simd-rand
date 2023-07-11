use rand_core::SeedableRng;
use shishua::Shishua;

fn main() {
    let mut rng = Shishua::seed_from_u64(0);

    for _ in 0..100_000_000 {
        println!("{}", rng.next_f64())
    }
}