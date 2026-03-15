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
pub fn seed_state(seed: u64) -> [u64; 3] {
    let mut splitmix_state = seed;
    let mut fast_loop = 0;
    let mut mix = 0;
    let mut loop_mix = 0;

    while fast_loop == 0 && mix == 0 && loop_mix == 0 {
        fast_loop = splitmix64_next(&mut splitmix_state);
        mix = splitmix64_next(&mut splitmix_state);
        loop_mix = splitmix64_next(&mut splitmix_state);
    }

    for _ in 0..WARMUP_ROUNDS {
        advance_state(&mut fast_loop, &mut mix, &mut loop_mix);
    }

    [fast_loop, mix, loop_mix]
}
