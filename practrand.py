#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "out"
DEFAULT_RNG_TEST = ROOT / "external/PractRand/PractRand/RNG_test"
DEFAULT_BINARY = ROOT / "target/release/examples/practrand"
DEFAULT_PRACTRAND_ARGS = ("stdin64", "-multithreaded")
BUILD_CMD = ("cargo", "+nightly", "build", "--example", "practrand", "--features", "portable", "--release")


@dataclass(frozen=True)
class Case:
    name: str
    rng: str


@dataclass(frozen=True)
class CaseGroup:
    name: str
    cases: tuple[Case, ...]


CASE_GROUPS = (
    CaseGroup(
        "biski64",
        (
            Case("scalar-biski64", "scalar-biski64"),
            Case("portable-biski64-x8", "portable-biski64-x8"),
            Case("specific-biski64-x8", "specific-biski64-x8"),
        ),
    ),
    CaseGroup(
        "frand",
        (
            Case("scalar-frand", "scalar-frand"),
            Case("portable-frand-x8", "portable-frand-x8"),
            Case("specific-frand-x8", "specific-frand-x8"),
        ),
    ),
    CaseGroup(
        "xoshiro256plus",
        (
            Case("scalar-xoshiro256plus", "scalar-xoshiro256plus"),
            Case("portable-xoshiro256plus-x8", "portable-xoshiro256plus-x8"),
            Case("specific-xoshiro256plus-x8", "specific-xoshiro256plus-x8"),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PractRand against the scalar and vectorized biski64/xoshiro256+/frand streams."
    )
    parser.add_argument("--seed", required=True, help="Seed passed through to the Rust practrand example.")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Root directory that will receive a per-run output folder."
    )
    parser.add_argument(
        "--rng-test",
        type=Path,
        default=DEFAULT_RNG_TEST,
        help="Path to the PractRand RNG_test binary. Assumes getpractrand has already been run.",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help="Path to the built Rust practrand example binary.",
    )
    parser.add_argument(
        "--practrand-arg",
        action="append",
        default=[],
        help="Argument forwarded to PractRand RNG_test. Repeat for multiple arguments.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip the cargo build step.")
    return parser.parse_args()


def build(binary: Path, out_dir: Path) -> None:
    build_log = out_dir / "build.log"

    with build_log.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(BUILD_CMD)}\n")
        log_file.write("\n")
        process = subprocess.Popen(
            BUILD_CMD,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise SystemExit(f"build failed; see {build_log}")

    ensure_executable(binary, "built practrand example")


def ensure_executable(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    if not os.access(path, os.X_OK):
        raise SystemExit(f"{label} is not executable: {path}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def make_run_dir(out_root: Path, seed: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"{timestamp}-seed-{seed.replace('/', '_')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def normalize_practrand_args(practrand_args: tuple[str, ...], seed: str) -> tuple[str, ...]:
    if "-seed" in practrand_args:
        return practrand_args
    return (*practrand_args, "-seed", seed)


def log_header(title: str) -> str:
    line = "=" * len(title)
    return f"{line}\n{title}\n{line}\n"


def run_case(
    case: Case,
    seed: str,
    binary: Path,
    rng_test: Path,
    practrand_args: tuple[str, ...],
    log_path: Path,
) -> tuple[str, float]:
    start = time.monotonic()
    status = "ok"

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(log_header(case.name))
        log_file.write(f"$ {binary} {case.rng} {seed}\n")
        log_file.write(f"$ {rng_test} {' '.join(practrand_args)}\n\n")

        producer = subprocess.Popen(
            [str(binary), case.rng, seed],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=log_file,
        )
        assert producer.stdout is not None

        consumer = subprocess.Popen(
            [str(rng_test), *practrand_args],
            cwd=ROOT,
            stdin=producer.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        producer.stdout.close()
        assert consumer.stdout is not None

        try:
            for line in consumer.stdout:
                sys.stdout.write(line)
                log_file.write(line)
        finally:
            consumer.stdout.close()

        consumer_code = consumer.wait()
        producer_code = producer.wait()

        if producer_code != 0 or consumer_code != 0:
            status = f"failed (generator={producer_code}, practrand={consumer_code})"

        log_file.write(f"\nstatus: {status}\n\n")

    return status, time.monotonic() - start


def run_group(
    group: CaseGroup,
    seed: str,
    binary: Path,
    rng_test: Path,
    practrand_args: tuple[str, ...],
    run_dir: Path,
) -> list[tuple[Case, str, float, Path]]:
    log_path = run_dir / f"{group.name}.log"
    log_path.write_text("", encoding="utf-8")

    results: list[tuple[Case, str, float, Path]] = []
    for case in group.cases:
        status, duration = run_case(case, seed, binary, rng_test, practrand_args, log_path)
        results.append((case, status, duration, log_path))
    return results


def write_summary(run_dir: Path, results: list[tuple[CaseGroup, list[tuple[Case, str, float, Path]]]]) -> None:
    summary_path = run_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as summary:
        for group, group_results in results:
            if not group_results:
                continue

            log_path = group_results[0][3]
            summary.write(f"{group.name}\n")
            summary.write(f"log: {display_path(log_path)}\n")
            summary.write("\n")

            for case, status, duration, _ in group_results:
                summary.write(f"{case.name}\n")
                summary.write(f"status: {status}\n")
                summary.write(f"duration_seconds: {duration:.2f}\n\n")


def main() -> int:
    args = parse_args()
    out_root = args.out_dir.resolve()
    rng_test = args.rng_test.resolve()
    binary = args.binary.resolve()
    practrand_args = normalize_practrand_args(tuple(args.practrand_arg) or DEFAULT_PRACTRAND_ARGS, args.seed)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(out_root, args.seed)

    ensure_executable(rng_test, "PractRand RNG_test")

    if not args.skip_build:
        build(binary, run_dir)
    else:
        ensure_executable(binary, "practrand example binary")

    results: list[tuple[CaseGroup, list[tuple[Case, str, float, Path]]]] = []

    try:
        print(f"output: {display_path(run_dir)}", flush=True)
        for group in CASE_GROUPS:
            print(f"==> {group.name}", flush=True)
            group_results = run_group(group, args.seed, binary, rng_test, practrand_args, run_dir)
            results.append((group, group_results))

            for case, status, duration, _ in group_results:
                print(f"  {case.name}: {status} ({duration:.2f}s)", flush=True)
            print(f"<== {group.name}", flush=True)
    finally:
        write_summary(run_dir, results)

    return 0 if all(status == "ok" for _, group_results in results for _, status, _, _ in group_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
