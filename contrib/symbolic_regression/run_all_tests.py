import os
import subprocess
import sys

TESTS = [
    "tests/test_newton.py",
    "tests/test_tully.py",
    "tests/test_leavitt.py",
    "tests/test_schechter.py",
    "tests/test_ideal_gas.py",
    "tests/test_planck.py",
    "tests/test_rydberg.py",
    "tests/test_hubble.py",
]


def main():
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    for test in TESTS:
        print(f"Running {test}...")
        result = subprocess.run([sys.executable, test], env=env)
        if result.returncode != 0:
            print(f"Failed: {test}")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
