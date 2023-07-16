#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod shishua {
    use rand_core::{RngCore, SeedableRng};
    use serial_test::serial;
    use simd_prng::specific::avx2::{Shishua, DEFAULT_BUFFER_SIZE};

    #[test]
    #[serial]
    fn deallocates() {
        let _profiler = dhat::Profiler::builder().testing().build();
        let start_stats = dhat::HeapStats::get();
        {
            let mut rng: Shishua<DEFAULT_BUFFER_SIZE> = Shishua::seed_from_u64(0);
            let n = rng.next_u32();
            assert!(n >= u32::MIN && n <= u32::MAX);

            let end_stats = dhat::HeapStats::get();
            dhat::assert_eq!(
                Shishua::<DEFAULT_BUFFER_SIZE>::LAYOUT.size(),
                end_stats.curr_bytes - start_stats.curr_bytes
            );
        }
        let stats = dhat::HeapStats::get();
        dhat::assert_eq!(start_stats.curr_bytes, stats.curr_bytes);
    }
}
