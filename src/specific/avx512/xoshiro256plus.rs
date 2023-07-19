use std::{
    arch::{x86_64::*, asm},
    mem::{self, transmute},
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx512::read_u64_into_vec;

use super::{simdprng::*, rotate_left};
use super::vecs::*;

pub struct Xoshiro256PlusX8Seed([u8; 256]);

impl Xoshiro256PlusX8Seed {
    pub fn new(seed: [u8; 256]) -> Self {
        Self(seed)
    }
}

impl Into<Xoshiro256PlusX8Seed> for [u8; 256] {
    fn into(self) -> Xoshiro256PlusX8Seed {
        Xoshiro256PlusX8Seed::new(self)
    }
}

impl Into<Xoshiro256PlusX8Seed> for Vec<u8> {
    fn into(self) -> Xoshiro256PlusX8Seed {
        assert!(self.len() == 256);
        Xoshiro256PlusX8Seed::new(self.try_into().unwrap())
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

#[repr(align(64))]
pub struct Xoshiro256PlusX8 {
    s0: __m512i,
    s1: __m512i,
    s2: __m512i,
    s3: __m512i,
}
impl Default for Xoshiro256PlusX8Seed {
    fn default() -> Xoshiro256PlusX8Seed {
        Xoshiro256PlusX8Seed([0; 256])
    }
}

impl AsMut<[u8]> for Xoshiro256PlusX8Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for Xoshiro256PlusX8 {
    type Seed = Xoshiro256PlusX8Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 8;
        const VECSIZE: usize = SIZE * LEN;
        // TODO: implement "jumps" between lanes?
        unsafe {
            let mut s0: __m512i = _mm512_setzero_si512();
            let mut s1: __m512i = _mm512_setzero_si512();
            let mut s2: __m512i = _mm512_setzero_si512();
            let mut s3: __m512i = _mm512_setzero_si512();
            read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)], &mut s0);
            read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)], &mut s1);
            read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)], &mut s2);
            read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)], &mut s3);

            Self { s0, s1, s2, s3 }
        }
    }
}

impl SimdPrng for Xoshiro256PlusX8 {
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_m512i(&mut self, vector: &mut __m512i) {
        unsafe {
            // const uint64_t result = s[0] + s[3];
            *vector = _mm512_add_epi64(self.s0, self.s3);

            // const uint64_t t = s[1] << 17;
            let t = _mm512_sll_epi64(self.s1, _mm_cvtsi32_si128(17));
        
            // s[2] ^= s[0];
            // s[3] ^= s[1];
            // s[1] ^= s[2];
            // s[0] ^= s[3];
            self.s2 = _mm512_xor_si512(self.s2, self.s0);
            self.s3 = _mm512_xor_si512(self.s3, self.s1);
            self.s1 = _mm512_xor_si512(self.s1, self.s2);
            self.s0 = _mm512_xor_si512(self.s0, self.s3);
        
            // s[2] ^= t;
            self.s2 = _mm512_xor_si512(self.s2, t);
        
            // s[3] = rotl(s[3], 45);
            self.s3 = rotate_left::<45>(self.s3);
        }
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_m512d(&mut self, result: &mut __m512d) {
        // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        unsafe {
            let mut v = _mm512_setzero_si512();
            self.next_m512i(&mut v);

            let lhs1 = _mm512_srl_epi64(v, _mm_cvtsi32_si128(11));
            let mut lhs2: __m512d;

            // this should be exposed through the '_mm512_cvtepu64_pd' C/C++ intrinsic,
            // but since I can't find this exposed in Rust anywhere,
            // we're doing it the inline asm way here
            // TODO: find out how to use the normal intrinsic from std::arch
            asm!(
                "vcvtuqq2pd {1}, {0}",
                in(zmm_reg) lhs1,
                out(zmm_reg) lhs2,
            );

            let rhs = _mm512_set1_pd(1.1102230246251565E-16);
            *result = _mm512_mul_pd(lhs2, rhs)
        }
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_u64x8(&mut self, vector: &mut U64x8) {
        unsafe {
            let mut v = _mm512_setzero_si512();
            self.next_m512i(&mut v);
            _mm512_store_epi64(transmute::<_, *mut i64>(vector), v);
        }
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_f64x8(&mut self, vector: &mut F64x8) {
        unsafe {
            let mut v = _mm512_setzero_pd();
            self.next_m512d(&mut v);
            _mm512_store_pd(vector.as_mut_ptr(), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use itertools::Itertools;
    use num_traits::PrimInt;
    use rand_core::{SeedableRng, RngCore};
    use serial_test::parallel;

    use crate::testutil::{test_uniform_distribution, DOUBLE_RANGE};

    use super::*;

    #[test]
    #[parallel]
    fn testing() {
        let mut rng: Xoshiro256PlusX8 = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: F64x8 = Default::default();

        rng.next_f64x8(&mut data);
        assert!(data.into_iter().all(|v| v != 0.0));
    }
}