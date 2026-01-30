use std::{
    arch::{asm, x86_64::*},
    mem::transmute,
};

use super::vecs::*;

pub trait SimdRand {
    fn next_m512i(&mut self) -> __m512i;

    #[inline(always)]
    fn next_m512d(&mut self) -> __m512d {
        unsafe {
            let v = self.next_m512i();

            let lhs = m512i_to_m512d(_mm512_srli_epi64::<11>(v));

            // PERF: This is precomputed based on the constants from the formula above
            // I found no other efficient (and succint) constant way of representing the RHS.
            // setzero and constants used in shifts like 11 above are automatically constant folded.
            // Writing out the actual formula ended up not being constant folded by the compiler.
            const RHS_FACTOR: [f64; 8] = [1.1102230246251565E-16; 8];
            const RHS: __m512d = unsafe { transmute::<[f64; 8], __m512d>(RHS_FACTOR) };

            _mm512_mul_pd(lhs, RHS)
        }
    }

    #[inline(always)]
    fn next_u64x8(&mut self) -> U64x8 {
        unsafe {
            let v = self.next_m512i();
            let mut vector = Default::default();
            _mm512_store_epi64(&mut vector as *mut U64x8 as *mut i64, v);
            vector
        }
    }

    #[inline(always)]
    fn next_f64x8(&mut self) -> F64x8 {
        unsafe {
            let v = self.next_m512d();
            let mut vector: F64x8 = Default::default();
            _mm512_store_pd(vector.as_mut_ptr(), v);
            vector
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
