#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![feature(portable_simd)]

use std::io::{ErrorKind, Write};

use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::*;
use simd_rand::portable::*;
use std::mem;

#[repr(align(64))]
struct Buf([u64; 512]);

type RngBaseline = Xoshiro256PlusPlus;
type RngVecSeed = Xoshiro256PlusPlusX8Seed;
type RngVecImpl = Xoshiro256PlusPlusX8;

const USE_SIMD: bool = true;

fn main() {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut seed: RngVecSeed = Default::default();
    rand::rng().fill_bytes(&mut *seed);
    let mut rng_baseline = RngBaseline::from_seed(seed[0..32].try_into().unwrap());
    let mut rng_simd = RngVecImpl::from_seed(seed);

    let mut buffer: Buf = Buf([0u64; 512]);
    let bytes = unsafe { mem::transmute::<&Buf, &[u8; mem::size_of::<Buf>()]>(&buffer) };

    loop {
        if USE_SIMD {
            // portable::Xoshiro256PlusPlusX8 succeeds 512 GB
            for i in (0..buffer.0.len()).step_by(8) {
                let values = rng_simd.next_u64x8();
                values.copy_to_slice(&mut buffer.0[i..i + 8]);
            }
        } else {
            // Xoshiro256++ baseline succeeds 512 GB
            for n in &mut buffer.0 {
                *n = rng_baseline.next_u64();
            }
        }

        match out.write_all(bytes) {
            Ok(()) => {}
            // Err(e) if e.kind() == ErrorKind::BrokenPipe => return,
            Err(e) => panic!("Failed: {e}"),
        };
    }
}
