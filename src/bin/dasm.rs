use rand::rngs::SmallRng;
use rand_core::{SeedableRng, RngCore};
use shishua::Shishua;

#[inline(never)]
fn do_shishua(rng: &mut Shishua) -> u64 {
    rng.next_u64()
}

#[inline(never)]
fn do_small_rng(rng: &mut SmallRng) -> u64 {
    rng.next_u64()
}

fn main() {
    let mut rng1: Shishua = Shishua::seed_from_u64(0);
    let mut rng2 = SmallRng::seed_from_u64(0);

    println!("{}", do_shishua(&mut rng1));
    println!("{}", do_small_rng(&mut rng2));
}