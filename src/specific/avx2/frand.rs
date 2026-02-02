use core::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx2::read_u64_into_vec;

use super::simdrand::*;

#[derive(Clone, Default)]
pub struct FrandX4Seed([u8; 32]);

impl FrandX4Seed {
    #[must_use]
    pub const fn new(seed: [u8; 32]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 32]> for FrandX4Seed {
    fn from(val: [u8; 32]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for FrandX4Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 32);
        let mut seed = [0u8; 32];
        seed.copy_from_slice(val);
        Self::new(seed)
    }
}

impl Deref for FrandX4Seed {
    type Target = [u8; 32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for FrandX4Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<[u8]> for FrandX4Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for FrandX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[repr(align(32))]
pub struct FrandX4 {
    seed: __m256i,
}

impl SeedableRng for FrandX4 {
    type Seed = FrandX4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 4;
        assert!(seed.len() == SIZE * LEN);

        let s = read_u64_into_vec(&seed[..]);

        Self { seed: s }
    }
}

/// 64-bit multiply using native AVX512DQ+VL instruction
#[cfg(all(target_arch = "x86_64", target_feature = "avx512dq", target_feature = "avx512vl"))]
#[inline(always)]
unsafe fn mullo_epi64(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_mullo_epi64(a, b) }
}

/// 64-bit multiply emulation for AVX2 without AVX512DQ
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512dq", target_feature = "avx512vl")))]
#[inline(always)]
unsafe fn mullo_epi64(a: __m256i, b: __m256i) -> __m256i {
    // a * b = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo) << 32
    let a_hi = _mm256_srli_epi64::<32>(a);
    let b_hi = _mm256_srli_epi64::<32>(b);

    let lo_lo = _mm256_mul_epu32(a, b);
    let a_lo_b_hi = _mm256_mul_epu32(a, b_hi);
    let a_hi_b_lo = _mm256_mul_epu32(a_hi, b);

    let cross = _mm256_add_epi64(a_lo_b_hi, a_hi_b_lo);
    let cross_shifted = _mm256_slli_epi64::<32>(cross);

    _mm256_add_epi64(lo_lo, cross_shifted)
}

impl SimdRand for FrandX4 {
    #[inline(always)]
    fn next_m256i(&mut self) -> __m256i {
        unsafe {
            let increment = _mm256_set1_epi64x(12964901029718341801_u64.cast_signed());
            let mul_xor = _mm256_set1_epi64x(149988720821803190_u64.cast_signed());

            let value = _mm256_add_epi64(self.seed, increment);
            self.seed = value;

            let xored = _mm256_xor_si256(mul_xor, value);
            let value = mullo_epi64(value, xored);

            _mm256_xor_si256(value, _mm256_srli_epi64::<32>(value))
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_FRAND_X4, test_uniform_distribution};

    use super::super::vecs::*;
    use super::*;

    type RngSeed = FrandX4Seed;
    type RngImpl = FrandX4;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_FRAND_X4.into();
        let mut rng = RngImpl::from_seed(seed);
        #[rustfmt::skip]
        let expected = [
            14822886784369691238_u64,
            16363031081693429805,
            853694022644052200,
            12415232109135804396,
            11186080767534476449,
            3540825981221545167,
            2242727420787380596,
            10506211554217625353,
            2569786415164250473,
            7749215101137833260,
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
}
