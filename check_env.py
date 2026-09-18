"""
Environment check — run this in Week 1, and any time something stops working.

    python check_env.py

Every line is a check you can fix on your own. Nothing here touches the network
or changes any of your files.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
    ("matplotlib", "matplotlib"),
]

PASS, FAIL, WARN = "  ok  ", " FAIL ", " warn "
results = []


def record(status: str, label: str, detail: str = "") -> None:
    results.append((status, label, detail))
    print(f"[{status}] {label}" + (f"\n         {detail}" if detail else ""))


def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) in {(3, 11), (3, 12)}:
        record(PASS, f"Python {version}")
    elif (major, minor) < (3, 11):
        record(FAIL, f"Python {version} is too old", "Install Python 3.11 and rebuild your virtual environment.")
    else:
        record(
            FAIL,
            f"Python {version} is newer than this project's pins",
            "requirements.txt pins numpy 1.26.4 and scikit-learn 1.5.2, which have no\n"
            "         wheels for this version — pip install will fail or build from source.\n"
            "         Install Python 3.11 and create the venv with it: python3.11 -m venv .venv",
        )


def check_virtualenv() -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        record(PASS, "running inside a virtual environment", sys.prefix)
    else:
        record(
            WARN,
            "not running inside a virtual environment",
            "Installing into your system Python works until two projects want different\n"
            "         versions of the same package. Create one:\n"
            "           python3.11 -m venv .venv && source .venv/bin/activate",
        )


def check_packages() -> None:
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
            continue
        version = getattr(module, "__version__", "?")
        record(PASS, f"{package_name} {version}")
    if missing:
        record(
            FAIL,
            f"missing packages: {', '.join(missing)}",
            "pip install -r requirements-dev.txt",
        )


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def check_git() -> None:
    if _git("rev-parse", "--git-dir") is None:
        record(FAIL, "this directory is not a git repository", "Did you clone your repo, or download a zip?")
        return

    remotes = _git("remote") or ""
    names = remotes.split()

    if "origin" in names:
        record(PASS, "git remote 'origin'", _git("remote", "get-url", "origin") or "")
    else:
        record(FAIL, "no git remote named 'origin'", "Clone your own repo rather than copying files around.")

    if "upstream" in names:
        record(PASS, "git remote 'upstream'", _git("remote", "get-url", "upstream") or "")
    else:
        record(
            WARN,
            "no git remote named 'upstream'",
            "Without it you cannot pull course updates. Add it:\n"
            "           git remote add upstream https://github.com/sivakarthik51/claims-cost-predictor-starter.git",
        )

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        record(PASS, f"on branch '{branch}'")


def check_seed() -> None:
    raw = os.environ.get("STUDENT_SEED")
    if raw is None:
        record(
            WARN,
            "STUDENT_SEED is not set — you are using the default cohort (42)",
            "Set your assigned seed so your dataset is your own:\n"
            "           export STUDENT_SEED=<your seed>\n"
            "         Add that line to your shell profile so it survives a new terminal.",
        )
        return
    try:
        record(PASS, f"STUDENT_SEED={int(raw)}")
    except ValueError:
        record(FAIL, f"STUDENT_SEED={raw!r} is not an integer")


def check_data() -> None:
    expected = ["members.csv", "claims.csv", "member_summary.csv"]
    present = [name for name in expected if (ROOT / "data" / name).exists()]
    if len(present) == len(expected):
        record(PASS, "data/ has all three CSVs")
    elif present:
        record(WARN, f"data/ is incomplete ({', '.join(present)})", "python generate_claims_data.py")
    else:
        record(
            WARN,
            "no data generated yet",
            "This is expected before you run: python generate_claims_data.py",
        )


def main() -> int:
    print("\nChecking your environment for the claims cost predictor project\n")
    check_python_version()
    check_virtualenv()
    check_packages()
    check_git()
    check_seed()
    check_data()

    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]

    print("\n" + "-" * 72)
    if failures:
        print(f"{len(failures)} check(s) failed. Fix those, then run this again.")
        print("Still stuck after trying the suggested fix? Bring the output of this")
        print("script to the session — it is exactly what someone needs to help you.")
        return 1

    if warnings:
        print(f"Everything essential works. {len(warnings)} thing(s) worth tidying up (see 'warn' above).")
    else:
        print("Everything checks out.")

    print("\nNext: python generate_claims_data.py && pytest tests/test_data.py")
    print("-" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
