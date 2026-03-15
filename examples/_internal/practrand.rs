#![cfg_attr(feature = "portable", feature(portable_simd))]

use core::str::FromStr;
use frand::Rand;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use simd_rand::portable::{
    FrandX4 as PortableFrandX4, FrandX8 as PortableFrandX8, SimdRandX4, SimdRandX8,
    Xoshiro256PlusX4 as PortableXoshiro256PlusX4, Xoshiro256PlusX8 as PortableXoshiro256PlusX8,
};
#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
use simd_rand::specific::avx2::{
    FrandX4 as SpecificFrandX4, SimdRand as SpecificSimdRandX4, Xoshiro256PlusX4 as SpecificXoshiro256PlusX4,
};
#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
use simd_rand::specific::avx512::{
    FrandX8 as SpecificFrandX8, SimdRand as SpecificSimdRandX8, Xoshiro256PlusX8 as SpecificXoshiro256PlusX8,
};
use std::io::{self, ErrorKind, Write};
use std::mem;
use std::process::ExitCode;

#[repr(align(64))]
struct Buf([u64; 512]);

#[derive(Clone, Copy)]
enum RngKind {
    ScalarFrand,
    ScalarXoshiro256Plus,
    PortableFrandX4,
    PortableFrandX8,
    PortableXoshiro256PlusX4,
    PortableXoshiro256PlusX8,
    #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
    SpecificFrandX4,
    #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
    SpecificXoshiro256PlusX4,
    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    SpecificFrandX8,
    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    SpecificXoshiro256PlusX8,
}

impl RngKind {
    const DEFAULT: Self = Self::PortableFrandX8;

    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "scalar-frand" => Ok(Self::ScalarFrand),
            "scalar-xoshiro256plus" => Ok(Self::ScalarXoshiro256Plus),
            "portable-frand-x4" => Ok(Self::PortableFrandX4),
            "portable-frand-x8" => Ok(Self::PortableFrandX8),
            "portable-xoshiro256plus-x4" => Ok(Self::PortableXoshiro256PlusX4),
            "portable-xoshiro256plus-x8" => Ok(Self::PortableXoshiro256PlusX8),
            #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
            "specific-frand-x4" => Ok(Self::SpecificFrandX4),
            #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
            "specific-xoshiro256plus-x4" => Ok(Self::SpecificXoshiro256PlusX4),
            #[cfg(not(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2")))]
            "specific-frand-x4" => Err(String::from("specific-frand-x4 is unavailable for this build")),
            #[cfg(not(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2")))]
            "specific-xoshiro256plus-x4" => {
                Err(String::from("specific-xoshiro256plus-x4 is unavailable for this build"))
            }
            #[cfg(all(
                feature = "specific",
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512dq",
                target_feature = "avx512vl"
            ))]
            "specific-frand-x8" => Ok(Self::SpecificFrandX8),
            #[cfg(all(
                feature = "specific",
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512dq",
                target_feature = "avx512vl"
            ))]
            "specific-xoshiro256plus-x8" => Ok(Self::SpecificXoshiro256PlusX8),
            #[cfg(not(all(
                feature = "specific",
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512dq",
                target_feature = "avx512vl"
            )))]
            "specific-frand-x8" => Err(String::from("specific-frand-x8 is unavailable for this build")),
            #[cfg(not(all(
                feature = "specific",
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512dq",
                target_feature = "avx512vl"
            )))]
            "specific-xoshiro256plus-x8" => {
                Err(String::from("specific-xoshiro256plus-x8 is unavailable for this build"))
            }
            _ => Err(format!("unknown RNG '{name}'")),
        }
    }
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [scalar-frand|scalar-xoshiro256plus|portable-frand-x4|portable-frand-x8|portable-xoshiro256plus-x4|portable-xoshiro256plus-x8|specific-frand-x4|specific-frand-x8|specific-xoshiro256plus-x4|specific-xoshiro256plus-x8] [seed]\n\
         seed may be decimal or 0x-prefixed hex"
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

fn fill_scalar_xoshiro256plus(rng: &mut Xoshiro256Plus, buffer: &mut [u64]) {
    for value in buffer {
        *value = rng.next_u64();
    }
}

fn fill_portable_frand_x4(rng: &mut PortableFrandX4, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(4) {
        rng.next_u64x4().copy_to_slice(chunk);
    }
}

fn fill_portable_frand_x8(rng: &mut PortableFrandX8, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(8) {
        rng.next_u64x8().copy_to_slice(chunk);
    }
}

fn fill_portable_xoshiro256plus_x4(rng: &mut PortableXoshiro256PlusX4, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(4) {
        rng.next_u64x4().copy_to_slice(chunk);
    }
}

fn fill_portable_xoshiro256plus_x8(rng: &mut PortableXoshiro256PlusX8, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(8) {
        rng.next_u64x8().copy_to_slice(chunk);
    }
}

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
fn fill_specific_frand_x4(rng: &mut SpecificFrandX4, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(4) {
        let values = rng.next_u64x4();
        chunk.copy_from_slice(&*values);
    }
}

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
fn fill_specific_xoshiro256plus_x4(rng: &mut SpecificXoshiro256PlusX4, buffer: &mut [u64]) {
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
fn fill_specific_frand_x8(rng: &mut SpecificFrandX8, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(8) {
        let values = rng.next_u64x8();
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
fn fill_specific_xoshiro256plus_x8(rng: &mut SpecificXoshiro256PlusX8, buffer: &mut [u64]) {
    for chunk in buffer.chunks_exact_mut(8) {
        let values = rng.next_u64x8();
        chunk.copy_from_slice(&*values);
    }
}

fn write_loop(mut fill: impl FnMut(&mut [u64]), out: &mut impl Write) -> io::Result<()> {
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

fn run(kind: RngKind, seed: u64, out: &mut impl Write) -> io::Result<()> {
    match kind {
        RngKind::ScalarFrand => {
            let mut rng = Rand::with_seed(seed);
            write_loop(|buffer| fill_scalar_frand(&mut rng, buffer), out)
        }
        RngKind::ScalarXoshiro256Plus => {
            let mut rng = Xoshiro256Plus::seed_from_u64(seed);
            write_loop(|buffer| fill_scalar_xoshiro256plus(&mut rng, buffer), out)
        }
        RngKind::PortableFrandX4 => {
            let mut rng = PortableFrandX4::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_frand_x4(&mut rng, buffer), out)
        }
        RngKind::PortableFrandX8 => {
            let mut rng = PortableFrandX8::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_frand_x8(&mut rng, buffer), out)
        }
        RngKind::PortableXoshiro256PlusX4 => {
            let mut rng = PortableXoshiro256PlusX4::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_xoshiro256plus_x4(&mut rng, buffer), out)
        }
        RngKind::PortableXoshiro256PlusX8 => {
            let mut rng = PortableXoshiro256PlusX8::seed_from_u64(seed);
            write_loop(|buffer| fill_portable_xoshiro256plus_x8(&mut rng, buffer), out)
        }
        #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
        RngKind::SpecificFrandX4 => {
            let mut rng = SpecificFrandX4::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_frand_x4(&mut rng, buffer), out)
        }
        #[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
        RngKind::SpecificXoshiro256PlusX4 => {
            let mut rng = SpecificXoshiro256PlusX4::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_xoshiro256plus_x4(&mut rng, buffer), out)
        }
        #[cfg(all(
            feature = "specific",
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        RngKind::SpecificFrandX8 => {
            let mut rng = SpecificFrandX8::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_frand_x8(&mut rng, buffer), out)
        }
        #[cfg(all(
            feature = "specific",
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        RngKind::SpecificXoshiro256PlusX8 => {
            let mut rng = SpecificXoshiro256PlusX8::seed_from_u64(seed);
            write_loop(|buffer| fill_specific_xoshiro256plus_x8(&mut rng, buffer), out)
        }
    }
}

fn try_main() -> Result<(), String> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| String::from("practrand"));
    let kind = args
        .next()
        .map_or(Ok(RngKind::DEFAULT), |raw| RngKind::parse(&raw))
        .map_err(|err| format!("{err}\n{}", usage(&program)))?;
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
