use std::{
    arch::x86_64::*,
    mem::{self, transmute},
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx2::read_u64_into_vec;

use super::{simdprng::*, rotate_left};
use super::vecs::*;

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

#[repr(align(32))]
pub struct Xoshiro256PlusX4 {
    s0: __m256i,
    s1: __m256i,
    s2: __m256i,
    s3: __m256i,
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

impl SeedableRng for Xoshiro256PlusX4 {
    type Seed = Xoshiro256PlusX4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 4;
        const VECSIZE: usize = SIZE * LEN;
        unsafe {
            let mut s0: __m256i = _mm256_setzero_si256();
            let mut s1: __m256i = _mm256_setzero_si256();
            let mut s2: __m256i = _mm256_setzero_si256();
            let mut s3: __m256i = _mm256_setzero_si256();
            read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)], &mut s0);
            read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)], &mut s1);
            read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)], &mut s2);
            read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)], &mut s3);

            Self { s0, s1, s2, s3 }
        }
    }
}

impl Xoshiro256PlusX4 {
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    pub fn next_m256d_pure_avx(&mut self, result: &mut __m256d) {
        // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        unsafe {
            let mut v = _mm256_setzero_si256();
            self.next_m256i(&mut v);

            let lhs1 = _mm256_srl_epi64(v, _mm_cvtsi32_si128(11));
            let lhs2 = super::u64_to_f64(lhs1);

            let rhs = _mm256_set1_pd(1.1102230246251565E-16);
            *result = _mm256_mul_pd(lhs2, rhs)
        }
    }
}

impl SimdPrng for Xoshiro256PlusX4 {
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_m256i(&mut self, vector: &mut __m256i) {
        unsafe {
            // const uint64_t result = s[0] + s[3];
            *vector = _mm256_add_epi64(self.s0, self.s3);

            // const uint64_t t = s[1] << 17;
            let t = _mm256_sll_epi64(self.s1, _mm_cvtsi32_si128(17));
        
            // s[2] ^= s[0];
            // s[3] ^= s[1];
            // s[1] ^= s[2];
            // s[0] ^= s[3];
            self.s2 = _mm256_xor_si256(self.s2, self.s0);
            self.s3 = _mm256_xor_si256(self.s3, self.s1);
            self.s1 = _mm256_xor_si256(self.s1, self.s2);
            self.s0 = _mm256_xor_si256(self.s0, self.s3);
        
            // s[2] ^= t;
            self.s2 = _mm256_xor_si256(self.s2, t);
        
            // s[3] = rotl(s[3], 45);
            self.s3 = rotate_left::<45>(self.s3);
        }
    }

    // Unfortunately this is not fast enough,
    // since there is no direct intrinsic for u64 -> f64 conversion (other than in avx512)
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_m256d(&mut self, result: &mut __m256d) {
        // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        unsafe {
            let mut v = Default::default();
            self.next_f64x4(&mut v);
            
            *result = _mm256_load_pd(v.as_ptr());
        }
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_u64x4(&mut self, vector: &mut U64x4) {
        unsafe {
            let mut v = _mm256_setzero_si256();
            self.next_m256i(&mut v);
            _mm256_store_si256(transmute::<_, *mut __m256i>(vector), v);
        }
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_f64x4(&mut self, vector: &mut F64x4) {
        let mut v = Default::default();
        self.next_u64x4(&mut v);

        vector[0] = (v[0] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[1] = (v[1] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[2] = (v[2] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[3] = (v[3] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    }
}