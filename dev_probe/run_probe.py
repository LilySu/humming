"""Compile-and-run driver for the dev_probe/*.cu files.

Mirrors WorldExplored's test_marlin_ldmatrix_s4_layout.py (vLLM PR #50096,
commit 3406be9f0d7): check nvcc >= 13.4, build for sm_90a, run, check stdout.
Not a pytest fixture on purpose — this isn't meant to be collected as a
standing test. Run directly:

    python3 dev_probe/run_probe.py dev_probe/ldmatrix_s4_layout.cu

To build and run wgmma_pairing_check.cu's documented negative control
(item 5b: proving the per-element check actually has discriminating power,
not just reporting PASS regardless of input), add --negative-control:

    python3 dev_probe/run_probe.py dev_probe/wgmma_pairing_check.cu --negative-control

This compiles with -DWRONG_FRAGMENT_ORDERING into a separate artifacts
directory and inverts the plain-run pass/fail interpretation: the run is a
PASS only if the built executable's own validate_wgmma() check correctly
reports FAILURE. compute-sanitizer must still report zero errors either
way — a wrong fragment mapping is a logic bug (wrong values), not an
out-of-bounds access, so memcheck passing tells you nothing about which
candidate is correct; only the plain run's per-element check does.

Per the review on this branch ("separate PTX acceptance from hardware
proof" / "keep Phase 0 closed only if it proves semantic layout, not
merely that literal PTX was emitted"), this driver does NOT stop at
"it ran and printed something." It also captures, into
dev_probe/artifacts/<probe-name>/, the artifacts needed to tell those two
things apart later:

  - the generated PTX (nvcc -ptx)
  - the cubin and its SASS disassembly (nvcc -cubin + cuobjdump)
  - register count / spill loads-stores (ptxas -v)
  - a metadata.txt recording the exact nvcc version, target arch, GPU
    (from nvidia-smi), and every command line run — so "what exactly did
    this pass against" is answered by a file, not a memory of the session.

Also runs the built executable under `compute-sanitizer --tool memcheck`
(if it's on PATH) and requires ITS exit code, not just the plain run's, for
overall success — an earlier ad hoc pattern on this branch invoked
compute-sanitizer with `|| true` inside multi-step shell scripts, which
keeps a script alive past a failing step but also means nothing downstream
can trust "the script exited 0" as "memcheck passed". This driver doesn't
make that tradeoff: main() returns non-zero if either the plain run or the
sanitizer run fails, and if compute-sanitizer isn't found at all, that's
treated as a failure (with a clear message) rather than silently skipped.

BUG (found by external review, fixed without needing a GPU to reproduce):
an earlier version of this driver checked compute-sanitizer's own exit
code but never passed `--error-exitcode`, and NVIDIA documents that
compute-sanitizer's exit code defaults to 0 EVEN WHEN IT DETECTS ERRORS
(https://docs.nvidia.com/compute-sanitizer/ — the --error-exitcode option
is exactly what makes a detected error propagate to the process exit
code). Without it, sanitizer_ok below could never actually be False from a
detected error — only from compute-sanitizer itself crashing or being
missing. Fixed by passing --error-exitcode explicitly; see
test_run_probe.py for a mocked-subprocess regression test of this exact
propagation path (no GPU needed to run that test).
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SANITIZER_ERROR_EXITCODE = 1


def nvcc_version_str(nvcc: str) -> str:
    result = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def nvcc_version(nvcc: str) -> tuple[int, int] | None:
    result = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    )
    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def gpu_info() -> str:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return "nvidia-smi not found"
    result = subprocess.run(
        [smi, "--query-gpu=name,driver_version,compute_cap", "--format=csv"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_logged(cmd: list[str], log: list[str], **kwargs) -> subprocess.CompletedProcess:
    log.append("$ " + " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def run_compute_sanitizer(compute_sanitizer: str, executable: Path) -> tuple[bool, subprocess.CompletedProcess]:
    """Run `executable` under compute-sanitizer --tool memcheck.

    Returns (sanitizer_ok, result). Pulled out of main() so the exact
    command line (in particular, --error-exitcode) and the returncode ->
    sanitizer_ok mapping can be unit-tested with a mocked subprocess.run,
    without needing nvcc or a GPU -- see test_run_probe.py.
    """
    result = subprocess.run(
        [
            compute_sanitizer,
            "--tool",
            "memcheck",
            "--error-exitcode",
            str(SANITIZER_ERROR_EXITCODE),
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    # compute-sanitizer's own exit code is the enforced signal here -- not a
    # text search for "ERROR SUMMARY: 0 errors" in its stdout, which is
    # exactly the kind of check that silently does nothing useful if the
    # output format ever changes. This is only actually true because
    # --error-exitcode is passed above: NVIDIA documents that without it,
    # the exit code defaults to 0 even when errors are detected (see this
    # file's module docstring for the bug this fixed).
    return result.returncode == 0, result


def decide_outcome(plain_returncode: int, sanitizer_ok: bool, negative_control: bool) -> tuple[int, str | None]:
    """The final pass/fail decision, isolated from all the subprocess/file-IO
    plumbing around it so it can be tested directly with plain ints/bools --
    no mocking needed at all for this half of the failure-propagation path.

    Returns (exit_code, failure_message_or_None).
    """
    if negative_control:
        # Inverted on purpose: this build is WRONG_FRAGMENT_ORDERING, a
        # known-bad fragment mapping (dev_probe/wgmma_pairing_check.cu
        # item 5). The addresses it computes still stay in-bounds (both
        # candidates only ever pick between two valid m_subgroup/k_half
        # values), so there is nothing for compute-sanitizer to catch --
        # this is a logic bug (wrong values matched to wrong positions),
        # not a memory-safety bug. The only thing that SHOULD catch it is
        # validate_wgmma()'s own per-element check, so a "pass" here means
        # that check correctly reported failure.
        if plain_returncode == 0:
            return 1, (
                "FAILED: negative-control build (WRONG_FRAGMENT_ORDERING) reported "
                "SUCCESS on its own validate_wgmma() check -- the check has no "
                "discriminating power, and a real regression in the fragment "
                "mapping could pass silently. See wgmma_pairing_check.cu's header "
                "comment on item 5's negative control."
            )
    elif plain_returncode != 0:
        return plain_returncode, "FAILED: plain run reported a non-zero exit code"

    if not sanitizer_ok:
        return 1, (
            "FAILED: compute-sanitizer did not report success -- a passing "
            "plain run does not satisfy this probe's gate on its own"
        )
    return 0, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="path to the probe .cu file")
    parser.add_argument(
        "--negative-control",
        action="store_true",
        help=(
            "build with -DWRONG_FRAGMENT_ORDERING (dev_probe/wgmma_pairing_check.cu's "
            "documented negative control) into a separate artifacts directory, and "
            "invert the plain-run pass/fail interpretation -- see this file's module "
            "docstring."
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"no such file: {source}", file=sys.stderr)
        return 2

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("nvcc not found on PATH", file=sys.stderr)
        return 1

    version = nvcc_version(nvcc)
    if version is None or version < (13, 4):
        print(f"nvcc {version} found; this probe requires CUDA 13.4+", file=sys.stderr)
        return 1

    # NOTE: "-arch=sm_90a" (nvcc's single-flag shorthand) was observed on
    # this toolkit to NOT propagate the "a" (architecture-specific-feature)
    # suffix through to the generated PTX's own .target directive -- ptxas
    # then rejected ldmatrix.s8.s4 with ".target 'sm_90'" (no "a") in the
    # error text, which is correctly-behaving ptxas rejecting a genuinely
    # different, narrower target than the one requested. The explicit
    # -gencode form is unambiguous about wanting the "a" variant on both
    # the virtual (compute_90a) and real (sm_90a) architecture, matching
    # vLLM PR #50096's own CMake gate (cuda_archs_sm90plus), which always
    # uses explicit a/f-suffixed arch strings rather than the bare form.
    gencode = "arch=compute_90a,code=sm_90a"
    # Negative-control builds get their own artifacts directory so they
    # never clobber the normal build's PTX/cubin/SASS/logs -- item 3's
    # branch-separation concern applies here too: two meaningfully
    # different builds of the same source must not share one output path.
    artifacts_name = source.stem + ("-wrong-fragment-ordering" if args.negative_control else "")
    artifacts_dir = Path(__file__).parent / "artifacts" / artifacts_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    command_log: list[str] = []
    defines = ["-DWRONG_FRAGMENT_ORDERING"] if args.negative_control else []

    executable = artifacts_dir / source.stem
    ptx_file = artifacts_dir / f"{source.stem}.ptx"
    cubin_file = artifacts_dir / f"{source.stem}.cubin"
    sass_file = artifacts_dir / f"{source.stem}.sass"
    ptxas_verbose_file = artifacts_dir / f"{source.stem}.ptxas_verbose.txt"
    metadata_file = artifacts_dir / "metadata.txt"

    print(f"compiling {source} for -gencode {gencode}{' with ' + ' '.join(defines) if defines else ''} ...")
    build = run_logged(
        [nvcc, "-std=c++17", "-O3", f"-gencode={gencode}", *defines, str(source), "-o", str(executable)],
        command_log,
    )
    if build.returncode != 0:
        print(build.stdout, build.stderr, file=sys.stderr)
        return build.returncode

    print("capturing PTX ...")
    run_logged([nvcc, f"-gencode={gencode}", *defines, "-ptx", str(source), "-o", str(ptx_file)], command_log)

    print("capturing cubin + SASS disassembly ...")
    run_logged([nvcc, f"-gencode={gencode}", *defines, "-cubin", str(source), "-o", str(cubin_file)], command_log)
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is not None and cubin_file.exists():
        sass = run_logged([cuobjdump, "--dump-sass", str(cubin_file)], command_log)
        sass_file.write_text(sass.stdout)
    else:
        sass_file.write_text("cuobjdump not found on PATH; SASS not captured\n")

    print("capturing register count / spill info (ptxas -v) ...")
    ptxas_verbose = run_logged(
        [nvcc, "-std=c++17", "-O3", f"-gencode={gencode}", *defines, "-Xptxas", "-v", str(source), "-o", str(executable)],
        command_log,
    )
    ptxas_verbose_file.write_text(ptxas_verbose.stdout + ptxas_verbose.stderr)

    metadata_file.write_text(
        "probe: {source}\n"
        "negative_control: {negctrl}\n"
        "nvcc: {nvcc_ver}\n"
        "target arch: {arch}\n"
        "gpu: {gpu}\n\n"
        "commands run:\n{cmds}\n".format(
            source=source,
            negctrl=args.negative_control,
            nvcc_ver=nvcc_version_str(nvcc),
            arch=gencode,
            gpu=gpu_info(),
            cmds="\n".join(command_log),
        )
    )

    print(f"running {executable} ...")
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    plain_run_file = artifacts_dir / "plain_run.txt"
    plain_run_file.write_text(
        f"exit code: {result.returncode}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n"
    )

    # Item 6 (dev_probe/NEXT_STEPS.md / humming-ldmatrix-s4-moe-plan.md §3.6):
    # run under memcheck too, and make its exit code an actual enforced
    # signal, not something read off printed "ERROR SUMMARY" text. Earlier
    # ad hoc scripts on this branch ran compute-sanitizer with `|| true` to
    # keep a multi-step shell script alive past a failing step -- fine for
    # an interactively-eyeballed run, but it means the same script's own
    # exit code can never be trusted as "did memcheck pass" by anything
    # unattended (a CI job, this driver, a future person skimming logs
    # instead of output). This driver does not swallow it: sanitizer_ok
    # below is required, in addition to the plain run, for main() to report
    # success.
    sanitizer_ok = True
    compute_sanitizer = shutil.which("compute-sanitizer")
    sanitizer_log_file = artifacts_dir / "compute_sanitizer.txt"
    if compute_sanitizer is None:
        sanitizer_log_file.write_text(
            "compute-sanitizer not found on PATH -- memcheck NOT run for this "
            "artifact. Do not treat a plain-run PASS alone as a substitute; "
            "re-run on an environment where compute-sanitizer is available "
            "before trusting this result.\n"
        )
        print(
            "WARNING: compute-sanitizer not found on PATH -- memcheck was NOT "
            "run. A plain-run pass alone does not satisfy this probe's "
            "correctness gate.",
            file=sys.stderr,
        )
        sanitizer_ok = False
    else:
        print(f"running {executable} under compute-sanitizer --tool memcheck ...")
        sanitizer_ok, sanitizer_result = run_compute_sanitizer(compute_sanitizer, executable)
        sanitizer_log_file.write_text(
            f"exit code: {sanitizer_result.returncode}\n\n"
            f"stdout:\n{sanitizer_result.stdout}\n\nstderr:\n{sanitizer_result.stderr}\n"
        )
        print(sanitizer_result.stdout, end="")
        if sanitizer_result.stderr:
            print(sanitizer_result.stderr, end="", file=sys.stderr)

    print(f"\nartifacts written to {artifacts_dir}/")

    exit_code, failure_message = decide_outcome(result.returncode, sanitizer_ok, args.negative_control)
    if failure_message is not None:
        print(f"{failure_message} (see {sanitizer_log_file} for the full sanitizer log)", file=sys.stderr)
    elif args.negative_control:
        print(
            "negative control correctly reported failure (validate_wgmma's own "
            f"exit code was {result.returncode}) -- the check has real "
            "discriminating power."
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
