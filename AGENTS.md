# Repository Guidelines for Agents

## Project Structure & Module Organization
- `src/` contains the library; `src/portable/` uses `std::simd`, `src/specific/{avx2,avx512}/` uses intrinsics.
- `tests/` holds integration tests (some are feature/arch gated).
- `benches/` contains Criterion benchmarks.
- `examples/_internal/` contains tooling examples (`dasm`, `profile`, `practrand`).
- `external/` is for third-party tools (e.g. PractRand); do not vendor code unless required.

## Build, Test, and Development Commands
- `make build`: nightly build with AVX512 flags and all features.
- `make test`: `cargo nextest run` for debug and release.
- `make lint`: nightly clippy for all targets/features.
- `make fmt`: rustfmt.
- `make bench`: Criterion benches with `portable` feature.
- `make check`: fast typecheck only.
- `make test-miri`: runs portable-only tests under miri (nightly) to catch UB.

Examples:
```
make lint
make test
make test-e2e
```
If you need `portable` SIMD or miri, use nightly (`cargo +nightly ...`). Confirm CPU feature flags before running AVX2/AVX512 code.

## Coding Style & Naming Conventions
- Rust 2024 edition; MSRV 1.89.
- `rustfmt` max width 120 (`rustfmt.toml`).
- Lints are strict: rust warnings deny; clippy all/pedantic/nursery/cargo deny.
- Use standard Rust naming (`snake_case` modules/functions, `UpperCamelCase` types).
- Unsafe is allowed only when justified by perf or intrinsics; document the reason.
