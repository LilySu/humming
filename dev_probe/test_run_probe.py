"""Regression tests for run_probe.py's failure-propagation logic.

External review finding (item 2): run_probe.py checked compute-sanitizer's
own exit code as its pass/fail signal, but never passed --error-exitcode.
NVIDIA documents that compute-sanitizer's exit code defaults to 0 EVEN WHEN
IT DETECTS ERRORS -- the option is what makes a detected error actually
propagate to the process exit code. Without it, a real memory error could
be silently reported as PASS by this driver.

This is pure Python with a mocked subprocess.run -- no nvcc, no GPU, no
compute-sanitizer binary needed to run these tests:

    python3 dev_probe/test_run_probe.py

Two things are tested, deliberately kept separate:
  1. run_compute_sanitizer() builds the right command line (in particular,
     that --error-exitcode is actually present) and maps a mocked
     CompletedProcess's returncode to sanitizer_ok correctly in both
     directions -- this is the part that would NOT have caught the
     original bug (the old code also correctly mapped returncode==0 to
     True; the bug was that a real detected error still produced
     returncode==0 upstream, which no amount of downstream mapping logic
     can fix -- that's why asserting --error-exitcode is actually in the
     command line matters as much as the returncode mapping).
  2. decide_outcome() -- the final exit-code decision -- covers all
     branches: plain run pass/fail, sanitizer ok/not, and negative-control
     mode's inverted interpretation.
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from run_probe import (  # noqa: E402
    SANITIZER_ERROR_EXITCODE,
    decide_outcome,
    run_compute_sanitizer,
)


def _completed(returncode: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["compute-sanitizer"], returncode=returncode, stdout="", stderr="")


def test_sanitizer_command_includes_error_exitcode():
    with mock.patch("run_probe.subprocess.run", return_value=_completed(0)) as mocked_run:
        run_compute_sanitizer("compute-sanitizer", Path("/fake/executable"))
    (cmd,), _kwargs = mocked_run.call_args
    assert "--error-exitcode" in cmd, f"--error-exitcode missing from command: {cmd}"
    flag_index = cmd.index("--error-exitcode")
    assert cmd[flag_index + 1] == str(SANITIZER_ERROR_EXITCODE), (
        f"--error-exitcode value is {cmd[flag_index + 1]!r}, expected {SANITIZER_ERROR_EXITCODE!r}"
    )
    print("[ok] compute-sanitizer command line includes --error-exitcode")


def test_sanitizer_ok_true_on_zero_returncode():
    with mock.patch("run_probe.subprocess.run", return_value=_completed(0)):
        sanitizer_ok, _result = run_compute_sanitizer("compute-sanitizer", Path("/fake/executable"))
    assert sanitizer_ok is True
    print("[ok] returncode 0 -> sanitizer_ok True")


def test_sanitizer_ok_false_on_nonzero_returncode():
    # Simulates compute-sanitizer actually detecting an error and (thanks to
    # --error-exitcode) propagating it -- the exact case the original bug
    # made unreachable in practice, since without that flag NVIDIA's own
    # compute-sanitizer would have returned 0 here regardless of what it
    # detected.
    with mock.patch("run_probe.subprocess.run", return_value=_completed(SANITIZER_ERROR_EXITCODE)):
        sanitizer_ok, _result = run_compute_sanitizer("compute-sanitizer", Path("/fake/executable"))
    assert sanitizer_ok is False
    print("[ok] nonzero returncode -> sanitizer_ok False")


def test_decide_outcome_normal_pass():
    exit_code, message = decide_outcome(plain_returncode=0, sanitizer_ok=True, negative_control=False)
    assert exit_code == 0 and message is None
    print("[ok] normal build: plain pass + sanitizer ok -> exit 0")


def test_decide_outcome_normal_plain_run_failure():
    exit_code, message = decide_outcome(plain_returncode=1, sanitizer_ok=True, negative_control=False)
    assert exit_code == 1 and message is not None
    print("[ok] normal build: plain run failure -> nonzero exit")


def test_decide_outcome_normal_sanitizer_failure():
    exit_code, message = decide_outcome(plain_returncode=0, sanitizer_ok=False, negative_control=False)
    assert exit_code != 0 and message is not None
    print("[ok] normal build: plain pass but sanitizer not ok -> nonzero exit")


def test_decide_outcome_negative_control_correctly_fails():
    # The negative-control build's validate_wgmma() SHOULD fail -- that's
    # the correct/desired outcome, so decide_outcome must report it as a
    # pass (exit 0) despite plain_returncode != 0.
    exit_code, message = decide_outcome(plain_returncode=1, sanitizer_ok=True, negative_control=True)
    assert exit_code == 0 and message is None
    print("[ok] negative control: validate_wgmma correctly failed -> exit 0 (inverted)")


def test_decide_outcome_negative_control_false_pass_is_caught():
    # If the negative-control build's validate_wgmma() reports SUCCESS,
    # that means the check has no discriminating power -- this must be
    # reported as a failure of the driver, not silently passed through.
    exit_code, message = decide_outcome(plain_returncode=0, sanitizer_ok=True, negative_control=True)
    assert exit_code != 0 and message is not None
    print("[ok] negative control: validate_wgmma false-passed -> caught as failure")


def test_decide_outcome_negative_control_still_requires_sanitizer_ok():
    # Even in negative-control mode (inverted plain-run interpretation),
    # a memory-safety error is still a real bug and must still fail.
    exit_code, message = decide_outcome(plain_returncode=1, sanitizer_ok=False, negative_control=True)
    assert exit_code != 0 and message is not None
    print("[ok] negative control: sanitizer failure still fails the run")


if __name__ == "__main__":
    test_sanitizer_command_includes_error_exitcode()
    test_sanitizer_ok_true_on_zero_returncode()
    test_sanitizer_ok_false_on_nonzero_returncode()
    test_decide_outcome_normal_pass()
    test_decide_outcome_normal_plain_run_failure()
    test_decide_outcome_normal_sanitizer_failure()
    test_decide_outcome_negative_control_correctly_fails()
    test_decide_outcome_negative_control_false_pass_is_caught()
    test_decide_outcome_negative_control_still_requires_sanitizer_ok()
    print("all run_probe.py failure-propagation checks passed")
