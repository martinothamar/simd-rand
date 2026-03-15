#![cfg_attr(feature = "portable", feature(portable_simd))]

use core::str::FromStr;
use frand::Rand;
use rand_core::{RngCore, SeedableRng};
use simd_rand::portable::{SimdRandX4 as PortableSimdRandX4, SimdRandX8 as PortableSimdRandX8};
#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
use simd_rand::specific::avx2::SimdRand as SpecificSimdRandX4;
#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
use simd_rand::specific::avx512::SimdRand as SpecificSimdRandX8;
use std::io::{self, ErrorKind, Write};
use std::mem;
use std::process::ExitCode;

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
type Shishua = simd_rand::specific::avx2::Shishua<{ simd_rand::specific::avx2::DEFAULT_BUFFER_SIZE }>;

#[repr(align(64))]
struct Buf([u64; 512]);

struct RngCase {
    name: &'static str,
    run: fn(u64, &mut dyn Write) -> io::Result<()>,
}

const RNG_CASES: &[RngCase] = &[
    RngCase {
        name: "scalar-biski64",
        run: |seed, out| {
            let mut rng = biski64::Biski64Rng::from_seed_for_stream(seed, 0, 1);
            write_loop(|buffer| fill_scalar_rngcore(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "scalar-frand",
        run: |seed, out| {
            let mut rng = frand::Rand::with_seed(seed);
            write_loop(|buffer| fill_scalar_frand(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "scalar-xoshiro256plus",
        run: |seed, out| {
            let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed);
            write_loop(|buffer| fill_scalar_rngcore(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "scalar-xoshiro256plusplus",
        run: |seed, out| {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            write_loop(|buffer| fill_scalar_rngcore(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-biski64-x4",
        run: |seed, out| {
            let mut rng = simd_rand::portable::Biski64X4::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x4(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-biski64-x8",
        run: |seed, out| {
            let mut rng = simd_rand::portable::Biski64X8::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x8(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-frand-x4",
        run: |seed, out| {
            let mut rng = simd_rand::portable::FrandX4::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x4(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-frand-x8",
        run: |seed, out| {
            let mut rng = simd_rand::portable::FrandX8::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x8(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-xoshiro256plus-x4",
        run: |seed, out| {
            let mut rng = simd_rand::portable::Xoshiro256PlusX4::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x4(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-xoshiro256plus-x8",
        run: |seed, out| {
            let mut rng = simd_rand::portable::Xoshiro256PlusX8::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x8(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-xoshiro256plusplus-x4",
        run: |seed, out| {
            let mut rng = simd_rand::portable::Xoshiro256PlusPlusX4::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x4(&mut rng, buffer), out)
        },
    },
    RngCase {
        name: "portable-xoshiro256plusplus-x8",
        run: |seed, out| {
            let mut rng = simd_rand::portable::Xoshiro256PlusPlusX8::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_x8(&mut rng, buffer), out)
        },
    },
    #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
    RngCase {
        name: "specific-biski64-x4",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx2::Biski64X4::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x4(&mut rng, buffer), out)
        },
    },
    #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
    RngCase {
        name: "specific-frand-x4",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx2::FrandX4::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x4(&mut rng, buffer), out)
        },
    },
    #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
    RngCase {
        name: "specific-xoshiro256plus-x4",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx2::Xoshiro256PlusX4::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x4(&mut rng, buffer), out)
        },
    },
    #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
    RngCase {
        name: "specific-xoshiro256plusplus-x4",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx2::Xoshiro256PlusPlusX4::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x4(&mut rng, buffer), out)
        },
    },
    #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
    RngCase {
        name: "specific-shishua-x4",
        run: |seed, out| {
            let mut rng = Shishua::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x4(&mut rng, buffer), out)
        },
    },
    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    RngCase {
        name: "specific-biski64-x8",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx512::Biski64X8::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x8(&mut rng, buffer), out)
        },
    },
    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    RngCase {
        name: "specific-frand-x8",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx512::FrandX8::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x8(&mut rng, buffer), out)
        },
    },
    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    RngCase {
        name: "specific-xoshiro256plus-x8",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx512::Xoshiro256PlusX8::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x8(&mut rng, buffer), out)
        },
    },
    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    RngCase {
        name: "specific-xoshiro256plusplus-x8",
        run: |seed, out| {
            let mut rng = simd_rand::specific::avx512::Xoshiro256PlusPlusX8::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_x8(&mut rng, buffer), out)
        },
    },
];
const DEFAULT_RNG: &str = "portable-frand-x8";

fn usage(program: &str) -> String {
    let names: Vec<_> = RNG_CASES.iter().map(|case| case.name).collect();
    format!(
        "usage: {program} [{}] [seed]\n\
         seed may be decimal or 0x-prefixed hex",
        names.join("|")
    )
}

fn parse_seed(raw: &str) -> Result<u64, String> {
    raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")).map_or_else(
        || u64::from_str(raw).map_err(|err| format!("invalid seed '{raw}': {err}")),
        |hex| u64::from_str_radix(hex, 16).map_err(|err| format!("invalid hex seed '{raw}': {err}")),
    )
}

fn fill_scalar_frand(rng: &mut Rand, buffer: &mut [u64]) {
    for value in buffer {
        *value = rng.r#gen::<u64>();
    }
}

fn fill_scalar_rngcore(rng: &mut impl RngCore, buffer: &mut [u64]) {
    for value in buffer {
        *value = rng.next_u64();
    }
}

fn fill_portable_x4(rng: &mut impl PortableSimdRandX4, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(4) {
        rng.next_u64x4().copy_to_slice(chunk);
    }
}

fn fill_portable_x8(rng: &mut impl PortableSimdRandX8, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(8) {
        rng.next_u64x8().copy_to_slice(chunk);
    }
}

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
fn fill_specific_x4(rng: &mut impl SpecificSimdRandX4, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(4) {
        let values = rng.next_u64x4();
        chunk.copy_from_slice(&*values);
    }
}

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
fn fill_specific_x8(rng: &mut impl SpecificSimdRandX8, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(8) {
        let values = rng.next_u64x8();
        chunk.copy_from_slice(&*values);
    }
}

fn write_loop(mut fill: impl FnMut(&mut [u64]), out: &mut dyn Write) -> io::Result<()> {
    let mut buffer = Buf([0; 512]);
    let bytes = unsafe { &*std::ptr::from_ref(&buffer).cast::<[u8; mem::size_of::<Buf>()]>() };

    loop {
        fill(&mut buffer.0);

        match out.write_all(bytes) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::BrokenPipe => return Ok(()),
            Err(err) => return Err(err),
        }
    }
}

fn run(case: &RngCase, seed: u64, out: &mut dyn Write) -> io::Result<()> {
    (case.run)(seed, out)
}

fn try_main() -> Result<(), String> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| String::from("practrand"));
    let kind = args.next().map_or_else(
        || {
            RNG_CASES
                .iter()
                .find(|case| case.name == DEFAULT_RNG)
                .ok_or_else(|| format!("default RNG '{DEFAULT_RNG}' is unavailable\n{}", usage(&program)))
        },
        |raw| {
            RNG_CASES
                .iter()
                .find(|case| case.name == raw)
                .ok_or_else(|| format!("unknown RNG '{raw}'\n{}", usage(&program)))
        },
    )?;
    let seed = args
        .next()
        .map_or(Ok(0), |raw| parse_seed(&raw))
        .map_err(|err| format!("{err}\n{}", usage(&program)))?;

    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument '{extra}'\n{}", usage(&program)));
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    run(kind, seed, &mut out).map_err(|err| format!("failed to write random stream: {err}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
