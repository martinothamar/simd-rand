#[cfg(all(test, feature = "portable"))]
use rand_core::RngCore;

pub const FAST_LOOP_INCREMENT: u64 = 0x9999999999999999;

const SPLITMIX_INCREMENT: u64 = 0x9e3779b97f4a7c15;
const SPLITMIX_MUL_0: u64 = 0xbf58476d1ce4e5b9;
const SPLITMIX_MUL_1: u64 = 0x94d049bb133111eb;
const WARMUP_ROUNDS: usize = 16;

#[inline(always)]
const fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INCREMENT);

    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL_0);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL_1);
    z ^ (z >> 31)
}

#[inline(always)]
const fn advance_state(fast_loop: &mut u64, mix: &mut u64, loop_mix: &mut u64) {
    let previous_fast_loop = *fast_loop;
    let previous_mix = *mix;

    *fast_loop = previous_fast_loop.wrapping_add(FAST_LOOP_INCREMENT);
    *mix = previous_mix.rotate_left(16).wrapping_add((*loop_mix).rotate_left(40));
    *loop_mix = previous_fast_loop ^ previous_mix;
}

#[must_use]
const fn seed_base_state(seed: u64) -> [u64; 3] {
    let mut splitmix_state = seed;
    let mut fast_loop = 0;
    let mut mix = 0;
    let mut loop_mix = 0;

    while fast_loop == 0 && mix == 0 && loop_mix == 0 {
        fast_loop = splitmix64_next(&mut splitmix_state);
        mix = splitmix64_next(&mut splitmix_state);
        loop_mix = splitmix64_next(&mut splitmix_state);
    }

    [fast_loop, mix, loop_mix]
}

fn warmup_state(mut state: [u64; 3]) -> [u64; 3] {
    for _ in 0..WARMUP_ROUNDS {
        let [fast_loop, mix, loop_mix] = &mut state;
        advance_state(fast_loop, mix, loop_mix);
    }

    state
}

#[must_use]
pub fn seed_state(seed: u64) -> [u64; 3] {
    warmup_state(seed_base_state(seed))
}

#[must_use]
pub fn seed_from_bytes(seed_bytes: &[u8]) -> u64 {
    let mut state = (seed_bytes.len() as u64) ^ FAST_LOOP_INCREMENT;
    let mut chunks = seed_bytes.chunks_exact(8);

    for chunk in &mut chunks {
        let mut word = [0; 8];
        word.copy_from_slice(chunk);
        state ^= u64::from_le_bytes(word);
        state = splitmix64_next(&mut state);
    }

    let remainder = chunks.remainder();
    if !remainder.is_empty() {
        let mut tail = [0; 8];
        tail[..remainder.len()].copy_from_slice(remainder);
        state ^= u64::from_le_bytes(tail);
        state = splitmix64_next(&mut state);
    }

    state
}

#[must_use]
pub fn seed_state_for_stream(seed: u64, stream_index: u64, total_streams: u64) -> [u64; 3] {
    assert!(total_streams >= 1);
    assert!(stream_index < total_streams);

    let [base_fast_loop, mix, loop_mix] = seed_base_state(seed);
    let fast_loop = if total_streams > 1 {
        let cycles_per_stream = u64::MAX / total_streams;
        let offset = stream_index
            .wrapping_mul(cycles_per_stream)
            .wrapping_mul(FAST_LOOP_INCREMENT);
        base_fast_loop.wrapping_add(offset)
    } else {
        base_fast_loop
    };

    warmup_state([fast_loop, mix, loop_mix])
}

#[must_use]
/// Matches biski64's upstream parallel-stream constructor; `from_seed` keeps
/// explicit raw lane control for callers that want independent per-lane seeds.
pub fn seed_stream_states<const LANES: usize>(seed: u64) -> [[u64; 3]; LANES] {
    assert!(LANES > 0);
    core::array::from_fn(|lane| seed_state_for_stream(seed, lane as u64, LANES as u64))
}

#[cfg(test)]
pub struct FixedBytesRng<const N: usize> {
    bytes: [u8; N],
    offset: usize,
}

#[cfg(test)]
impl<const N: usize> FixedBytesRng<N> {
    pub const fn new(bytes: [u8; N]) -> Self {
        Self { bytes, offset: 0 }
    }
}

#[cfg(test)]
impl<const N: usize> rand_core::RngCore for FixedBytesRng<N> {
    fn next_u32(&mut self) -> u32 {
        rand_core::impls::next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_fill(self)
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let end = self.offset + dst.len();
        assert!(end <= self.bytes.len());
        dst.copy_from_slice(&self.bytes[self.offset..end]);
        self.offset = end;
    }
}

#[cfg(all(test, feature = "portable"))]
pub fn reference_sequence<const N: usize>(seed: u64) -> [u64; N] {
    let mut rng = biski64::Biski64Rng::from_seed_for_stream(seed, 0, 1);
    let mut output = [0; N];

    for value in &mut output {
        *value = rng.next_u64();
    }

    output
}

#[cfg(all(test, feature = "portable"))]
pub fn parallel_reference_vectors<const LANES: usize, const N: usize>(seed: u64) -> [[u64; LANES]; N] {
    let mut rngs: [biski64::Biski64Rng; LANES] =
        core::array::from_fn(|lane| biski64::Biski64Rng::from_seed_for_stream(seed, lane as u64, LANES as u64));
    let mut output = [[0; LANES]; N];

    for row in &mut output {
        for (value, rng) in row.iter_mut().zip(&mut rngs) {
            *value = rng.next_u64();
        }
    }

    output
}

#[cfg(test)]
pub fn assert_rngs_match<const LANES: usize, R>(mut lhs: R, mut rhs: R, mut next: impl FnMut(&mut R) -> [u64; LANES]) {
    for _ in 0..3 {
        assert_eq!(next(&mut lhs), next(&mut rhs));
    }
}

#[cfg(all(test, feature = "portable"))]
pub fn assert_seed_from_u64_matches_parallel_streams<const LANES: usize, R>(
    seed: u64,
    mut rng: R,
    mut next: impl FnMut(&mut R) -> [u64; LANES],
) {
    for expected in parallel_reference_vectors::<LANES, 10>(seed) {
        assert_eq!(next(&mut rng), expected);
    }
}

#[cfg(all(test, feature = "portable"))]
pub fn assert_from_rng_matches_parallel_streams<const LANES: usize, const BYTES: usize, R>(
    seed: [u8; BYTES],
    from_rng: impl FnOnce(&mut FixedBytesRng<BYTES>) -> R,
    mut next: impl FnMut(&mut R) -> [u64; LANES],
) {
    let mut rng = from_rng(&mut FixedBytesRng::new(seed));
    let master_seed = seed_from_bytes(&seed);

    for expected in parallel_reference_vectors::<LANES, 10>(master_seed) {
        assert_eq!(next(&mut rng), expected);
    }
}

#[cfg(test)]
mod tests {
    use rand_core::{RngCore, TryRngCore};

    use super::{
        FAST_LOOP_INCREMENT, FixedBytesRng, seed_from_bytes, seed_state, seed_state_for_stream, seed_stream_states,
    };

    #[test]
    fn seed_from_bytes_handles_remainder() {
        let full_words = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let with_tail = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];

        assert_ne!(seed_from_bytes(&full_words), seed_from_bytes(&with_tail));
        assert_eq!(seed_from_bytes(&[]), FAST_LOOP_INCREMENT);
    }

    #[test]
    fn single_stream_matches_seed_state() {
        assert_eq!(seed_state_for_stream(42, 0, 1), seed_state(42));
    }

    #[test]
    fn seed_stream_states_matches_per_stream_seeding() {
        let states = seed_stream_states::<4>(42);

        for (lane, state) in states.into_iter().enumerate() {
            assert_eq!(state, seed_state_for_stream(42, lane as u64, 4));
        }
    }

    #[test]
    fn fixed_bytes_rng_reads_u32_u64_and_fill_bytes_in_order() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut rng = FixedBytesRng::new(bytes);

        assert_eq!(rng.next_u32(), u32::from_le_bytes([1, 2, 3, 4]));
        assert_eq!(rng.next_u64(), u64::from_le_bytes([5, 6, 7, 8, 9, 10, 11, 12]));

        let mut more = [0u8; 4];
        let mut rng = FixedBytesRng::new(bytes);
        rng.try_fill_bytes(&mut more).unwrap();
        assert_eq!(more, [1, 2, 3, 4]);
    }
}
