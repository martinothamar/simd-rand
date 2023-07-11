use rand_core::{SeedableRng};
use shishua::Shishua;

fn main() {
    let mut rng1 = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
    let mut rng2 = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

    for _ in 0..10 {
        let value1 = rng1.next_f32();
        let value2 = rng2.next_f64();
        println!("{:.6}", value1);
        println!("{:.6}", value2);
        println!("--------");
    }
}