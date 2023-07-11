use rand::Rng;
use rand_core::SeedableRng;
use shishua::Shishua;

fn main() {
    let mut rng: Shishua = Shishua::seed_from_u64(0);

    for _ in 0..10 {
        println!("{}", rng.gen_range(0.0..1.0))
    }
}