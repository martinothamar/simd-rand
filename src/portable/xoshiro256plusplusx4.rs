use std::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x4,
};

use rand_core::SeedableRng;

use super::{SimdRandX4, read_u64_into_vec, rotate_left};

#[derive(Clone)]
pub struct Xoshiro256PlusPlusX4Seed([u8; 128]);

impl Xoshiro256PlusPlusX4Seed {
    pub fn new(seed: [u8; 128]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 128]> for Xoshiro256PlusPlusX4Seed {
    fn from(val: [u8; 128]) -> Self {
        Xoshiro256PlusPlusX4Seed::new(val)
    }
}

impl From<Vec<u8>> for Xoshiro256PlusPlusX4Seed {
    fn from(val: Vec<u8>) -> Self {
        assert!(val.len() == 128);
        Xoshiro256PlusPlusX4Seed::new(val.try_into().unwrap())
    }
}

impl Deref for Xoshiro256PlusPlusX4Seed {
    type Target = [u8; 128];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Xoshiro256PlusPlusX4Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for Xoshiro256PlusPlusX4Seed {
    fn default() -> Xoshiro256PlusPlusX4Seed {
        Xoshiro256PlusPlusX4Seed([0; 128])
    }
}

impl AsRef<[u8]> for Xoshiro256PlusPlusX4Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Xoshiro256PlusPlusX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

pub struct Xoshiro256PlusPlusX4 {
    s0: u64x4,
    s1: u64x4,
    s2: u64x4,
    s3: u64x4,
}

impl SeedableRng for Xoshiro256PlusPlusX4 {
    type Seed = Xoshiro256PlusPlusX4Seed;

    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = u64x4::LEN;
        const VECSIZE: usize = SIZE * LEN;

        let s0 = read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)]);
        let s1 = read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)]);
        let s2 = read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)]);
        let s3 = read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)]);

        Self { s0, s1, s2, s3 }
    }
}

impl SimdRandX4 for Xoshiro256PlusPlusX4 {
    fn next_u64x4(&mut self) -> u64x4 {
        let result = rotate_left(self.s0 + self.s3, 23) + self.s0;

        let t = self.s1 << u64x4::splat(17);

        self.s2 ^= self.s0;
        self.s3 ^= self.s1;
        self.s1 ^= self.s2;
        self.s0 ^= self.s3;

        self.s2 ^= t;

        self.s3 = rotate_left(self.s3, 45);

        result
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};
    use std::simd::*;

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_256, test_uniform_distribution};

    use super::*;

    type RngSeed = Xoshiro256PlusPlusX4Seed;
    type RngImpl = Xoshiro256PlusPlusX4;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_256.into();
        let mut rng = RngImpl::from_seed(seed);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro256plusplus.c
        #[rustfmt::skip]
        let expected = [
            41943041, 58720359, 3588806011781223, 3591011842654386,
            9228616714210784205, 9973669472204895162, 14011001112246962877,
            12406186145184390807, 15849039046786891736, 10450023813501588000,
        ];
        for e in expected {
            let mem = rng.next_u64x4();
            for &v in mem.as_array().iter() {
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    fn sample_u64x4() {
        let mut seed: RngSeed = Default::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_u64x4().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = *rng.next_u64x4().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    fn sample_f64x4() {
        let mut seed: RngSeed = Default::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_f64x4().as_array();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let values = *rng.next_f64x4().as_array();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    #[cfg_attr(miri, ignore)]
    fn sample_f64x4_distribution() {
        let mut seed: RngSeed = Default::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<f64x4> = None;
        let mut current_index: usize = 0;

        test_uniform_distribution::<10_000_000, f64>(
            || match &current {
                Some(vector) if current_index < 4 => {
                    let result = vector[current_index];
                    current_index += 1;
                    result
                }
                _ => {
                    current_index = 0;
                    let vector = rng.next_f64x4();
                    let result = vector[current_index];
                    current = Some(vector);
                    current_index += 1;
                    result
                }
            },
            DOUBLE_RANGE,
        );
    }
}
