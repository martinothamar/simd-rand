use core::arch::x86_64::*;

pub use frand::*;
pub use shishua::*;
pub use simdrand::*;
pub use vecs::*;
pub use xoshiro256plus::*;
pub use xoshiro256plusplus::*;

mod frand;
mod shishua;
mod simdrand;
mod vecs;
mod xoshiro256plus;
mod xoshiro256plusplus;

#[inline(always)]
fn read_u64_into_vec(src: &[u8]) -> __m256i {
    assert!(src.len() == core::mem::size_of::<__m256i>());

    // This intrinsic is specifically the unaligned load variant.
    #[allow(clippy::cast_ptr_alignment)]
    unsafe {
        _mm256_loadu_si256(src.as_ptr().cast::<__m256i>())
    }
}

#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
fn rotate_left<const K: i32>(x: __m256i) -> __m256i {
    // rotl: (x << k) | (x >> (64 - k))
    unsafe { _mm256_rol_epi64::<K>(x) }
}

#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl")))]
fn rotate_left<const K: i32>(x: __m256i) -> __m256i {
    // rotl: (x << k) | (x >> (64 - k))
    unsafe {
        let left = _mm256_sll_epi64(x, _mm_cvtsi32_si128(K));
        let right = _mm256_srl_epi64(x, _mm_cvtsi32_si128(64 - K));
        _mm256_or_si256(left, right)
    }
}

#[cfg(test)]
mod tests {
    use core::arch::x86_64::{__m256i, _mm256_store_si256};

    use super::{read_u64_into_vec, vecs::U64x4};

    #[test]
    fn read_u64_into_vec_preserves_lane_order() {
        let expected: [u64; 4] = [
            0x0123_4567_89AB_CDEFu64,
            0x1112_1314_1516_1718u64,
            0x2122_2324_2526_2728u64,
            0x3132_3334_3536_3738u64,
        ];
        let mut src = [0u8; 32];

        for (index, value) in expected.into_iter().enumerate() {
            src[(index * 8)..((index + 1) * 8)].copy_from_slice(&value.to_le_bytes());
        }

        let vector = read_u64_into_vec(&src);
        let mut actual = U64x4::default();

        unsafe {
            _mm256_store_si256(core::ptr::from_mut(&mut actual).cast::<__m256i>(), vector);
        }

        assert_eq!(&*actual, &expected);
    }
}
