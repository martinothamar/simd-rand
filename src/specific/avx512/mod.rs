use core::{arch::x86_64::*, mem};

pub use frand::*;
// pub use shishua::*;
pub use simdrand::*;
pub use vecs::*;
pub use xoshiro256plus::*;
pub use xoshiro256plusplus::*;

mod frand;
// mod shishua;
mod simdrand;
mod vecs;
mod xoshiro256plus;
mod xoshiro256plusplus;

#[allow(clippy::identity_op, clippy::erasing_op)]
#[inline(always)]
fn read_u64_into_vec(src: &[u8]) -> __m512i {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 8);
    unsafe {
        _mm512_set_epi64(
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 7)..(SIZE * 8)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 6)..(SIZE * 7)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 5)..(SIZE * 6)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 4)..(SIZE * 5)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap())),
        )
    }
}

#[cfg(test)]
mod tests {
    use core::arch::x86_64::_mm512_store_epi64;

    use super::{read_u64_into_vec, vecs::U64x8};

    #[test]
    fn read_u64_into_vec_preserves_lane_order() {
        let expected: [u64; 8] = [
            0x0123_4567_89AB_CDEFu64,
            0x1112_1314_1516_1718u64,
            0x2122_2324_2526_2728u64,
            0x3132_3334_3536_3738u64,
            0x4142_4344_4546_4748u64,
            0x5152_5354_5556_5758u64,
            0x6162_6364_6566_6768u64,
            0x7172_7374_7576_7778u64,
        ];
        let mut src = [0u8; 64];

        for (index, value) in expected.into_iter().enumerate() {
            src[(index * 8)..((index + 1) * 8)].copy_from_slice(&value.to_le_bytes());
        }

        let vector = read_u64_into_vec(&src);
        let mut actual = U64x8::default();

        unsafe {
            _mm512_store_epi64(core::ptr::from_mut(&mut actual).cast::<i64>(), vector);
        }

        assert_eq!(&*actual, &expected);
    }
}
