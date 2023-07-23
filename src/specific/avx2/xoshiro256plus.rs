use std::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx2::read_u64_into_vec;

use super::{rotate_left, simdrand::*};

pub struct Xoshiro256PlusX4Seed([u8; 128]);

impl Xoshiro256PlusX4Seed {
    pub fn new(seed: [u8; 128]) -> Self {
        Self(seed)
    }
}

impl Into<Xoshiro256PlusX4Seed> for [u8; 128] {
    fn into(self) -> Xoshiro256PlusX4Seed {
        Xoshiro256PlusX4Seed::new(self)
    }
}

impl Into<Xoshiro256PlusX4Seed> for Vec<u8> {
    fn into(self) -> Xoshiro256PlusX4Seed {
        assert!(self.len() == 128);
        Xoshiro256PlusX4Seed::new(self.try_into().unwrap())
    }
}

impl Deref for Xoshiro256PlusX4Seed {
    type Target = [u8; 128];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Xoshiro256PlusX4Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for Xoshiro256PlusX4Seed {
    fn default() -> Xoshiro256PlusX4Seed {
        Xoshiro256PlusX4Seed([0; 128])
    }
}

impl AsMut<[u8]> for Xoshiro256PlusX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[repr(align(32))]
pub struct Xoshiro256PlusX4 {
    s0: __m256i,
    s1: __m256i,
    s2: __m256i,
    s3: __m256i,
}

impl SeedableRng for Xoshiro256PlusX4 {
    type Seed = Xoshiro256PlusX4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 4;
        const VECSIZE: usize = SIZE * LEN;
        
        let s0 = read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)]);
        let s1 = read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)]);
        let s2 = read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)]);
        let s3 = read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)]);

        Self { s0, s1, s2, s3 }
    }
}

impl SimdRand for Xoshiro256PlusX4 {
    #[inline(always)]
    fn next_m256i(&mut self) -> __m256i {
        unsafe {
            let vector = _mm256_add_epi64(self.s0, self.s3);

            let t = _mm256_sll_epi64(self.s1, _mm_cvtsi32_si128(17));

            self.s2 = _mm256_xor_si256(self.s2, self.s0);
            self.s3 = _mm256_xor_si256(self.s3, self.s1);
            self.s1 = _mm256_xor_si256(self.s1, self.s2);
            self.s0 = _mm256_xor_si256(self.s0, self.s3);

            self.s2 = _mm256_xor_si256(self.s2, t);

            self.s3 = rotate_left::<45>(self.s3);

            vector
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};
    use serial_test::parallel;

    use crate::testutil::{test_uniform_distribution, DOUBLE_RANGE, REF_SEED_256};

    use super::super::vecs::*;
    use super::*;

    type RngSeed = Xoshiro256PlusX4Seed;
    type RngImpl = Xoshiro256PlusX4;

    #[test]
    #[parallel]
    fn reference() {
        let seed: RngSeed = REF_SEED_256.into();
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
        for &e in &expected {
            let mem = rng.next_u64x4();
            for v in mem.into_iter() {
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    #[parallel]
    fn sample_u64x4() {
        let mut seed: RngSeed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = rng.next_u64x4();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = rng.next_u64x4();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x4() {
        let mut seed: RngSeed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = rng.next_f64x4();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let values = rng.next_f64x4();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x4_distribution() {
        let mut seed: RngSeed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<F64x4> = None;
        let mut current_index: usize = 0;

        test_uniform_distribution::<10_000_000, f64>(
            || match &current {
                Some(vector) if current_index < 4 => {
                    let result = vector[current_index];
                    current_index += 1;
                    return result;
                }
                _ => {
                    current_index = 0;
                    let vector = rng.next_f64x4();
                    let result = vector[current_index];
                    current = Some(vector);
                    current_index += 1;
                    return result;
                }
            },
            DOUBLE_RANGE,
        );
    }
}
