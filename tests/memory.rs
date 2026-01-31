#![cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use rand_core::{RngCore, SeedableRng};
use simd_rand::specific::avx2::{DEFAULT_BUFFER_SIZE, Shishua};

fn assert_allocation_and_deallocation<const N: usize>() {
    let start_stats = dhat::HeapStats::get();
    {
        let mut rng: Shishua<N> = Shishua::seed_from_u64(0);
        let _n = rng.next_u32();

        let end_stats = dhat::HeapStats::get();
        dhat::assert_eq!(
            Shishua::<N>::LAYOUT.size(),
            end_stats.curr_bytes - start_stats.curr_bytes
        );
    }
    let stats = dhat::HeapStats::get();
    dhat::assert_eq!(start_stats.curr_bytes, stats.curr_bytes);
}

#[test]
fn deallocates() {
    let _profiler = dhat::Profiler::builder().testing().build();
    assert_allocation_and_deallocation::<DEFAULT_BUFFER_SIZE>();
    assert_allocation_and_deallocation::<256>();
    assert_allocation_and_deallocation::<1024>();
}
