use std::{
    arch::x86_64::*,
    mem::{self, transmute}, ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use super::vecs::*;
use super::simdprng::*;

pub struct Xoshiro256PlusPlusX4Seed([u8; 128]);

impl Xoshiro256PlusPlusX4Seed {
    pub fn new(seed: [u8; 128]) -> Self {
        Self(seed)
    }
}

impl Into<Xoshiro256PlusPlusX4Seed> for [u8; 128] {
    fn into(self) -> Xoshiro256PlusPlusX4Seed {
        Xoshiro256PlusPlusX4Seed::new(self)
    }
}

impl Into<Xoshiro256PlusPlusX4Seed> for Vec<u8> {
    fn into(self) -> Xoshiro256PlusPlusX4Seed {
        assert!(self.len() == 128);
        Xoshiro256PlusPlusX4Seed::new(self.try_into().unwrap())
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
    fn default() -> Xoshiro256PlusPlusX4Seed {
        Xoshiro256PlusPlusX4Seed([0; 128])
    }
}

impl AsMut<[u8]> for Xoshiro256PlusPlusX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for Xoshiro256PlusPlusX4 {
    type Seed = Xoshiro256PlusPlusX4Seed;

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

#[inline]
fn read_u64_into_vec(src: &[u8], dst: &mut __m256i) {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 4);
    unsafe {
        *dst = _mm256_set_epi64x(
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap(),
            )),
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap(),
            )),
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap(),
            )),
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap(),
            )),
        )
    }
}

impl SimdPrng for Xoshiro256PlusPlusX4 {
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_m256i(&mut self, vector: &mut __m256i) {
        unsafe {
            let s0 = _mm256_load_si256(transmute::<_, *const __m256i>(&self.s0));
            let s3 = _mm256_load_si256(transmute::<_, *const __m256i>(&self.s3));

            // s[0] + s[3]
            let sadd = _mm256_add_epi64(s0, s3);

            // rotl(s[0] + s[3], 23)
            // rotl: (x << k) | (x >> (64 - k)), k = 23
            let rotl = rotate_left::<23>(sadd);

            // rotl(...) + s[0]
            *vector = _mm256_add_epi64(rotl, s0);

            //         let t = self.s[1] << 17;
            let s1 = _mm256_load_si256(transmute::<_, *const __m256i>(&self.s1));
            let t = _mm256_sll_epi64(s1, _mm_cvtsi32_si128(17));

            //         self.s[2] ^= self.s[0];
            //         self.s[3] ^= self.s[1];
            //         self.s[1] ^= self.s[2];
            //         self.s[0] ^= self.s[3];
            self.s2 = _mm256_xor_si256(self.s2, self.s0);
            self.s3 = _mm256_xor_si256(self.s3, self.s1);
            self.s1 = _mm256_xor_si256(self.s1, self.s2);
            self.s0 = _mm256_xor_si256(self.s0, self.s3);

            //         self.s[2] ^= t;
            self.s2 = _mm256_xor_si256(self.s2, t);

            //         self.s[3] = self.s[3].rotate_left(45);
            self.s3 = rotate_left::<45>(self.s3);
        }
    }

    // Unfortunately this is not fast enough,
    // since there is no direct intrinsic for u64 -> f64 conversion (other than in avx512)
    // #[cfg_attr(dasm, inline(never))]
    // #[cfg_attr(not(dasm), inline(always))]
    // fn next_m256d(&mut self, result: &mut __m256d) {
    //     // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    //     unsafe {
    //         let mut v = _mm256_set1_epi64x(0);
    //         self.next_m256i(&mut v);

    //         let lhs1 = _mm256_srl_epi64(v, _mm_cvtsi32_si128(11));
    //         let lhs2 = u64_to_f64(lhs1);

    //         let rhs = _mm256_set1_pd(1.1102230246251565E-16);
    //         *result = _mm256_mul_pd(lhs2, rhs)
    //     }
    // }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_u64x4(&mut self, vector: &mut U64x4) {
        unsafe {
            let mut v = _mm256_set1_epi64x(0);
            self.next_m256i(&mut v);
            _mm256_store_si256(transmute::<_, *mut __m256i>(vector), v);
        }
    }

    // Unfortunately this is not fast enough,
    // since there is no direct intrinsic for u64 -> f64 conversion (other than in avx512)
    // #[cfg_attr(dasm, inline(never))]
    // #[cfg_attr(not(dasm), inline(always))]
    // pub fn next_f64x4(&mut self, mem: &mut F64x4) {

    //     unsafe {
    //         let mut v = _mm256_set1_pd(0.0);
    //         self.next_m256d(&mut v);
    //         _mm256_store_pd(transmute::<_, *mut f64>(&mut mem.0), v);
    //     }
    // }

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

#[cfg_attr(dasm, inline(never))]
#[cfg_attr(not(dasm), inline(always))]
fn rotate_left<const K: i32>(x: __m256i) -> __m256i {
    unsafe {
        // rotl: (x << k) | (x >> (64 - k)), k = 23
        let rotl = _mm256_sll_epi64(x, _mm_cvtsi32_si128(K));
        _mm256_or_si256(rotl, _mm256_srl_epi64(x, _mm_cvtsi32_si128(64 - K)))
    }
}

// No direct conv intrinsic in AVX2, this hack is from
// https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
// Unfortunately not faster than just loading and operating on the scalars
// #[cfg_attr(dasm, inline(never))]
// #[cfg_attr(not(dasm), inline(always))]
// unsafe fn u64_to_f64(v: __m256i) -> __m256d {
//     let magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);
//     let magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);
//     let magic_i_all = _mm256_set1_epi64x(0x4530000000100000);
//     let magic_d_all = _mm256_castsi256_pd(magic_i_all);

//     let v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
//     let v_hi = _mm256_srli_epi64(v, 32);
//     let v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);
//     let v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
//     let result = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
//     result
// }

#[cfg(test)]
mod tests {
    use std::mem;

    use itertools::Itertools;
    use num_traits::PrimInt;
    use rand::rngs::SmallRng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    #[test]
    fn reference() {
        let ref_seed: [u8; 128] = [
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0, 0, 
            2, 0, 0, 0, 0, 0, 0, 0, 
            2, 0, 0, 0, 0, 0, 0, 0, 
            2, 0, 0, 0, 0, 0, 0, 0, 
            3, 0, 0, 0, 0, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0,
            4, 0, 0, 0, 0, 0, 0, 0,
            4, 0, 0, 0, 0, 0, 0, 0,
            4, 0, 0, 0, 0, 0, 0, 0,
            4, 0, 0, 0, 0, 0, 0, 0,
        ];
        let seed: Xoshiro256PlusPlusX4Seed = ref_seed.into();
        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed.try_into().unwrap());
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro256plusplus.c
        let expected = [
            41943041,
            58720359,
            3588806011781223,
            3591011842654386,
            9228616714210784205,
            9973669472204895162,
            14011001112246962877,
            12406186145184390807,
            15849039046786891736,
            10450023813501588000,
        ];
        for &e in &expected {
            let mut mem = Default::default();
            rng.next_u64x4(&mut mem);
            for v in mem.into_iter() {    
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    fn generate_vector_u64() {
        let mut seeder = SmallRng::seed_from_u64(0);

        let mut seed: Xoshiro256PlusPlusX4Seed = Default::default();
        seeder.fill_bytes(&mut seed[..]);

        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed);

        let mut values = U64x4::new([0; 4]);
        rng.next_u64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    fn generate_vector_f64() {
        let mut seeder = SmallRng::seed_from_u64(0);

        let mut seed: Xoshiro256PlusPlusX4Seed = Default::default();
        seeder.fill_bytes(&mut seed[..]);

        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed);

        let mut values = F64x4::new([0.0; 4]);
        rng.next_f64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    fn bitfiddling() {
        let v = 0b00000000_00000000_00000000_000000001u32;
        print(v);
    }

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
