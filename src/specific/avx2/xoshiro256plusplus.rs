use std::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx2::read_u64_into_vec;

use super::{rotate_left, simdrand::*};

#[derive(Clone)]
pub struct Xoshiro256PlusPlusX4Seed([u8; 128]);

impl Xoshiro256PlusPlusX4Seed {
    #[must_use]
    pub const fn new(seed: [u8; 128]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 128]> for Xoshiro256PlusPlusX4Seed {
    fn from(val: [u8; 128]) -> Self {
        Self::new(val)
    }
}

impl From<Vec<u8>> for Xoshiro256PlusPlusX4Seed {
    fn from(val: Vec<u8>) -> Self {
        assert!(val.len() == 128);
        Self::new(val.try_into().unwrap())
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

#[repr(align(32))]
pub struct Xoshiro256PlusPlusX4 {
    s0: __m256i,
    s1: __m256i,
    s2: __m256i,
    s3: __m256i,
}
impl Default for Xoshiro256PlusPlusX4Seed {
    fn default() -> Self {
        Self([0; 128])
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

impl SeedableRng for Xoshiro256PlusPlusX4 {
    type Seed = Xoshiro256PlusPlusX4Seed;

    #[allow(clippy::identity_op, clippy::erasing_op)]
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

impl SimdRand for Xoshiro256PlusPlusX4 {
    #[inline(always)]
    fn next_m256i(&mut self) -> __m256i {
        unsafe {
            let vector = _mm256_add_epi64(rotate_left::<23>(_mm256_add_epi64(self.s0, self.s3)), self.s0);

            let t = _mm256_slli_epi64::<17>(self.s1);

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
    use std::mem;

    use itertools::Itertools;
    use num_traits::PrimInt;
    use rand_core::{RngCore, SeedableRng};

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_256, test_uniform_distribution};

    use super::super::vecs::*;
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
        for &e in &expected {
            let mem = rng.next_u64x4();
            for v in &*mem {
                assert_eq!(*v, e);
            }
        }
    }

    #[test]
    fn sample_u64x4() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
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
    fn sample_f64x4() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = rng.next_f64x4();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let values = rng.next_f64x4();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "distribution test requires release mode")]
    fn sample_f64x4_distribution() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<F64x4> = None;
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

    #[test]
    fn bitfiddling() {
        let v = 0b0000_0000_0000_0000_0000_0000_0000_0001_u32;
        print(v);
    }

    #[allow(clippy::items_after_statements)]
    fn print<T>(v: T)
    where
        T: PrimInt + ToString,
    {
        let size = mem::size_of::<T>();
        let bit_size = size * 8;

        const PREFIX: &str = "0b";

        let mut output = String::with_capacity(PREFIX.len() + bit_size + (size - 1));
        output.push_str(PREFIX);
        let one = T::one();
        for n in (0..bit_size).rev() {
            let bit = (v >> n) & one;
            output.push_str(&bit.to_string());
            if n != 0 && n % 8 == 0 {
                output.push('_');
            }
        }
        println!("{output}");
    }
}
