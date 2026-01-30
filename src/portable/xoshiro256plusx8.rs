use std::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x8,
};

use rand_core::SeedableRng;

use super::{SimdRandX8, read_u64_into_vec, rotate_left};

#[derive(Clone)]
pub struct Xoshiro256PlusX8Seed([u8; 256]);

impl Xoshiro256PlusX8Seed {
    pub fn new(seed: [u8; 256]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 256]> for Xoshiro256PlusX8Seed {
    fn from(val: [u8; 256]) -> Self {
        Xoshiro256PlusX8Seed::new(val)
    }
}

impl From<Vec<u8>> for Xoshiro256PlusX8Seed {
    fn from(val: Vec<u8>) -> Self {
        assert!(val.len() == 256);
        Xoshiro256PlusX8Seed::new(val.try_into().unwrap())
    }
}

impl Deref for Xoshiro256PlusX8Seed {
    type Target = [u8; 256];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Xoshiro256PlusX8Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for Xoshiro256PlusX8Seed {
    fn default() -> Xoshiro256PlusX8Seed {
        Xoshiro256PlusX8Seed([0; 256])
    }
}

impl AsRef<[u8]> for Xoshiro256PlusX8Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Xoshiro256PlusX8Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

pub struct Xoshiro256PlusX8 {
    s0: u64x8,
    s1: u64x8,
    s2: u64x8,
    s3: u64x8,
}

impl SeedableRng for Xoshiro256PlusX8 {
    type Seed = Xoshiro256PlusX8Seed;

    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = u64x8::LEN;
        const VECSIZE: usize = SIZE * LEN;

        let s0 = read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)]);
        let s1 = read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)]);
        let s2 = read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)]);
        let s3 = read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)]);

        Self { s0, s1, s2, s3 }
    }
}

impl SimdRandX8 for Xoshiro256PlusX8 {
    fn next_u64x8(&mut self) -> u64x8 {
        let result = self.s0 + self.s3;

        let t = self.s1 << u64x8::splat(17);

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

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_512, test_uniform_distribution};

    use super::*;

    type RngSeed = Xoshiro256PlusX8Seed;
    type RngImpl = Xoshiro256PlusX8;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_512.into();
        let mut rng = RngImpl::from_seed(seed);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro256plus.c
        #[rustfmt::skip]
        let expected = [
            5,
            211106232532999,
            211106635186183,
            9223759065350669058,
            9250833439874351877,
            13862484359527728515,
            2346507365006083650,
            1168864526675804870,
            34095955243042024,
            3466914240207415127,
        ];
        for e in expected {
            let mem = rng.next_u64x8();
            for &v in mem.as_array().iter() {
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    fn sample_u64x8() {
        let mut seed: RngSeed = Default::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_u64x8().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = *rng.next_u64x8().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    fn sample_f64x4() {
        let mut seed: RngSeed = Default::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_f64x8().as_array();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let values = *rng.next_f64x8().as_array();

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

        let mut current: Option<f64x8> = None;
        let mut current_index: usize = 0;

        test_uniform_distribution::<10_000_000, f64>(
            || match &current {
                Some(vector) if current_index < 8 => {
                    let result = vector[current_index];
                    current_index += 1;
                    result
                }
                _ => {
                    current_index = 0;
                    let vector = rng.next_f64x8();
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
