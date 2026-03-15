const HASH_MUL: u64 = 4997996261773036203;

#[must_use]
pub const fn hash_seed(seed: u64) -> u64 {
    let seed = (seed ^ (seed >> 32)).wrapping_mul(HASH_MUL);
    let seed = (seed ^ (seed >> 32)).wrapping_mul(HASH_MUL);
    seed ^ (seed >> 32)
}

#[must_use]
pub fn repeated_seed_bytes<const BYTES: usize>(seed: u64) -> [u8; BYTES] {
    let mut bytes = [0u8; BYTES];

    for chunk in bytes.chunks_exact_mut(8) {
        chunk.copy_from_slice(&seed.to_le_bytes());
    }

    bytes
}

#[must_use]
pub fn hash_seed_bytes<const BYTES: usize>(seed: &[u8]) -> [u8; BYTES] {
    let mut bytes = [0u8; BYTES];
    assert_eq!(seed.len(), BYTES);

    for (src, dst) in seed.chunks_exact(8).zip(bytes.chunks_exact_mut(8)) {
        let mut word = [0u8; 8];
        word.copy_from_slice(src);
        dst.copy_from_slice(&hash_seed(u64::from_le_bytes(word)).to_le_bytes());
    }

    bytes
}

#[cfg(test)]
pub mod test_support {
    const REFERENCE_STEPS: usize = if cfg!(miri) { 32 } else { 1024 };

    pub fn ref_seed_x4() -> [u8; 32] {
        super::repeated_seed_bytes::<32>(1)
    }

    #[cfg(any(
        feature = "portable",
        all(
            feature = "specific",
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        )
    ))]
    pub fn ref_seed_x8() -> [u8; 64] {
        super::repeated_seed_bytes::<64>(1)
    }

    pub fn assert_seed_from_u64_matches_upstream<const LANES: usize, R>(
        seed: u64,
        mut rng: R,
        mut next: impl FnMut(&mut R) -> [u64; LANES],
    ) {
        let mut reference = ::frand::Rand::with_seed(seed);

        for _ in 0..REFERENCE_STEPS {
            assert_eq!(next(&mut rng), [reference.r#gen::<u64>(); LANES]);
        }
    }
}
