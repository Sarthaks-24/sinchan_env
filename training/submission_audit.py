"""
Submission readiness audit for OpenEnv hackathon packaging.

Usage:
    py -3 training/submission_audit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OPENENV_YAML = ROOT / "openenv.yaml"
NOTEBOOK = ROOT / "training" / "ShinChan_GRPO_Training.ipynb"
TRAIN_SCRIPT = ROOT / "training" / "train_sinchan.py"
PRE_FLIGHT = ROOT / "training" / "preflight_space.py"

REQUIRED_EVIDENCE = [
    ROOT / "assets" / "reward_curve_total.png",
    ROOT / "assets" / "baseline_comparison.png",
]
OPTIONAL_EVIDENCE = [
    ROOT / "assets" / "loss_curve.png",
    ROOT / "training" / "artifacts" / "eval_summary.json",
]


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def has_url(text: str, needle: str) -> bool:
    return needle in text


def main() -> int:
    hard_fail = False

    if README.exists():
        ok("README.md exists")
        readme_text = README.read_text(encoding="utf-8", errors="replace")
    else:
        fail("README.md missing")
        return 1

    for path in [OPENENV_YAML, NOTEBOOK, TRAIN_SCRIPT, PRE_FLIGHT]:
        if path.exists():
            ok(f"{path.relative_to(ROOT)} exists")
        else:
            fail(f"{path.relative_to(ROOT)} missing")
            hard_fail = True

    if has_url(readme_text, "https://huggingface.co/spaces/Gladiator-codes/sinchan-env"):
        ok("README includes Hugging Face Space card URL")
    else:
        fail("README missing Hugging Face Space card URL")
        hard_fail = True

    if has_url(readme_text, "colab.research.google.com"):
        ok("README includes Colab notebook URL")
    else:
        fail("README missing Colab notebook URL")
        hard_fail = True

    if "TODO_ADD_PUBLIC_URL" in readme_text:
        warn("Mini-blog/video URL still TODO_ADD_PUBLIC_URL")
    elif re.search(r"https?://", readme_text):
        ok("README contains at least one additional public URL")
    else:
        warn("README has no public blog/video URL")

    missing_required = [p for p in REQUIRED_EVIDENCE if not p.exists()]
    if missing_required:
        for p in missing_required:
            warn(f"Missing required evidence file: {p.relative_to(ROOT)}")
    else:
        ok("Required evidence plots present")

    missing_optional = [p for p in OPTIONAL_EVIDENCE if not p.exists()]
    for p in missing_optional:
        warn(f"Optional evidence missing: {p.relative_to(ROOT)}")
    for p in OPTIONAL_EVIDENCE:
        if p.exists():
            ok(f"Optional evidence present: {p.relative_to(ROOT)}")

    print()
    if hard_fail:
        fail("Submission audit failed due to hard requirements.")
        return 1

    ok("Submission audit completed (warnings may remain).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

