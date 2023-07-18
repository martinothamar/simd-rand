use std::{
    arch::x86_64::*,
    mem::{self, transmute},
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx2::read_u64_into_vec;

use super::{simdprng::*, rotate_left};
use super::vecs::*;

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

impl Xoshiro256PlusPlusX4 {
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

impl SimdPrng for Xoshiro256PlusPlusX4 {
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_m256i(&mut self, vector: &mut __m256i) {
        unsafe {
            // const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
            *vector = _mm256_add_epi64(rotate_left::<23>(_mm256_add_epi64(self.s0, self.s3)), self.s0);

            // let t = self.s[1] << 17;
            let t = _mm256_sll_epi64(self.s1, _mm_cvtsi32_si128(17));

            // self.s[2] ^= self.s[0];
            // self.s[3] ^= self.s[1];
            // self.s[1] ^= self.s[2];
            // self.s[0] ^= self.s[3];
            self.s2 = _mm256_xor_si256(self.s2, self.s0);
            self.s3 = _mm256_xor_si256(self.s3, self.s1);
            self.s1 = _mm256_xor_si256(self.s1, self.s2);
            self.s0 = _mm256_xor_si256(self.s0, self.s3);

            // self.s[2] ^= t;
            self.s2 = _mm256_xor_si256(self.s2, t);

            // self.s[3] = self.s[3].rotate_left(45);
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
    fn reference() {
        let seed: Xoshiro256PlusPlusX4Seed = REF_SEED.into();
        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro256plusplus.c
        #[rustfmt::skip]
        let expected = [
            41943041, 58720359, 3588806011781223, 3591011842654386,
            9228616714210784205, 9973669472204895162, 14011001112246962877,
            12406186145184390807, 15849039046786891736, 10450023813501588000,
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
    #[parallel]
    fn sample_u64x4() {
        let mut seed: Xoshiro256PlusPlusX4Seed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed);

        let mut values = U64x4::new([0; 4]);
        rng.next_u64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let mut values = U64x4::new([0; 4]);
        rng.next_u64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x4() {
        let mut seed: Xoshiro256PlusPlusX4Seed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed);

        let mut values = F64x4::new([0.0; 4]);
        rng.next_f64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let mut values = F64x4::new([0.0; 4]);
        rng.next_f64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x4_distribution() {
        let mut seed: Xoshiro256PlusPlusX4Seed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed);

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
                    let mut vector = Default::default();
                    current_index = 0;
                    rng.next_f64x4(&mut vector);
                    let result = vector[current_index];
                    current = Some(vector);
                    current_index += 1;
                    return result;
                }
            },
            DOUBLE_RANGE,
        );
    }

    #[test]
    #[parallel]
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
    
    #[rustfmt::skip]
    const REF_SEED: [u8; 128] = [
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
}
