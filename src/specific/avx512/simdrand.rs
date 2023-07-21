use std::{
    arch::{asm, x86_64::*},
    mem::transmute,
};

use super::vecs::*;

pub trait SimdRand {
    fn next_m512i(&mut self, vector: &mut __m512i);

    #[inline(always)]
    fn next_m512d(&mut self, result: &mut __m512d) {
        // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        unsafe {
            let mut v = _mm512_setzero_si512();
            self.next_m512i(&mut v);

            let lhs = m512i_to_m512d(_mm512_srl_epi64(v, _mm_cvtsi32_si128(11)));

            // PERF: This is precomputed based on the constants from the formula above
            // I found no other efficient (and succint) constant way of representing the RHS.
            // setzero and constants used in shifts like 11 above are automatically constant folded.
            // Writing out the actual formula ended up not being constant folded by the compiler.
            const RHS_FACTOR: [f64; 8] = [1.1102230246251565E-16; 8];
            const RHS: __m512d = unsafe { transmute::<[f64; 8], __m512d>(RHS_FACTOR) };

            *result = _mm512_mul_pd(lhs, RHS)
        }
    }

    #[inline(always)]
    fn next_u64x8(&mut self, vector: &mut U64x8) {
        unsafe {
            let mut v = _mm512_setzero_si512();
            self.next_m512i(&mut v);
            _mm512_store_epi64(transmute::<_, *mut i64>(vector), v);
        }
    }

    #[inline(always)]
    fn next_f64x8(&mut self, vector: &mut F64x8) {
        unsafe {
            let mut v = _mm512_setzero_pd();
            self.next_m512d(&mut v);
            _mm512_store_pd(vector.as_mut_ptr(), v);
        }
    }
}

#[inline(always)]
unsafe fn m512i_to_m512d(src: __m512i) -> __m512d {
    // this should be exposed through the '_mm512_cvtepu64_pd' C/C++ intrinsic,
    // but since I can't find this exposed in Rust anywhere,
    // we're doing it the inline asm way here
    // TODO: find out what happened in std::arch
    let mut dst: __m512d;
    asm!(
        "vcvtuqq2pd {1}, {0}",
        in(zmm_reg) src,
        out(zmm_reg) dst,
        // PERF: 'nostack' tells the Rust compiler that our asm won't touch the stack.
        // If we don't include this, the compiler might inject additional
        // instructions to make the stack pointer 16byte aligned in accordance to x64 ABI.
        // If we were to 'call' in our inline asm, it would have to push the 8byte return address
        // onto the stack, so the stack would have to be 16byte aligned before this happened
        options(nostack),
    );
    dst
}
