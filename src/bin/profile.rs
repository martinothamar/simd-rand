use rand::Rng;
use rand_core::SeedableRng;
use shishua::Shishua;

fn main() {
    let mut rng1: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
    let mut rng2: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

    for _ in 0..10 {
        let value1 = rng1.gen_range(0.0..1.0);
        let value2 = rng2.gen_range(0.0..1.0);
        println!("{:.6}", value1);
        println!("{:.6}", value2);
        println!("--------");
    }
}