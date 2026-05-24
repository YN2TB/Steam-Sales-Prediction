"""harness/check.py — gate trước commit."""

import argparse
import subprocess
import sys

FULL = [
    ("Format", ["python", "-m", "ruff", "format", "--check", "."]),
    ("Lint", ["python", "-m", "ruff", "check", "."]),
    ("Eval", ["python", "-m", "pytest", "harness/eval.py", "-q", "--tb=short"]),
]
QUICK = [
    ("Format", ["python", "-m", "ruff", "format", "--check", "."]),
    ("Lint", ["python", "-m", "ruff", "check", "."]),
]


def run(checks):
    failed = []
    for name, cmd in checks:
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  {'✅' if r.returncode == 0 else '❌'} {name}")
        if r.returncode != 0:
            out = (r.stdout + r.stderr).strip()
            if out:
                print(f"     {out[:300]}")
            failed.append(name)
    print()
    if failed:
        print(f"⛔ Blocked — failed: {', '.join(failed)}\n")
        return False
    print("✅ All checks passed.\n")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    print("\n🔍 Steam Sales Window — Harness Check\n" + "─" * 40)
    sys.exit(0 if run(QUICK if args.quick else FULL) else 1)
