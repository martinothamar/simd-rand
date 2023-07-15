use rand::rngs::SmallRng;
use rand_core::{SeedableRng, RngCore};
use shishua::{Shishua, xoshiro256plusplus::Xoshiro256PlusPlusX4};

#[inline(never)]
fn do_shishua(rng: &mut Shishua) -> u64 {
    rng.next_u64()
}

#[inline(never)]
fn do_small_rng(rng: &mut SmallRng) -> u64 {
    rng.next_u64()
}

#[inline(never)]
fn do_xoshiro_x4(rng: &mut Xoshiro256PlusPlusX4) -> u64 {
    let mut result = Default::default();
    rng.next_u64x4(&mut result);
    result.0[0]
}

fn main() {
    let mut rng1: Shishua = Shishua::seed_from_u64(0);
    let mut rng2 = SmallRng::seed_from_u64(0);
    let mut rng3 = Xoshiro256PlusPlusX4::seed_from_u64(0);

    println!("{}", do_shishua(&mut rng1));
    println!("{}", do_small_rng(&mut rng2));
    println!("{}", do_xoshiro_x4(&mut rng3));
}