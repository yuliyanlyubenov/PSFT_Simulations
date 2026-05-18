"""Smoke tests: every example script runs to completion without error."""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
EX = os.path.join(SIM, "examples")


class TestExamplesRun(unittest.TestCase):
    pass


def _make_case(filename):
    def _test(self):
        path = os.path.join(EX, filename)
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=120, cwd=SIM,
        )
        self.assertEqual(result.returncode, 0,
                         msg=f"{filename} failed:\nSTDOUT:\n{result.stdout}\n"
                             f"STDERR:\n{result.stderr}")
        self.assertTrue(len(result.stdout) > 50,
                        msg=f"{filename} produced suspiciously little output.")
    _test.__name__ = f"test_{os.path.splitext(filename)[0]}"
    return _test


for f in sorted(os.listdir(EX)):
    # Match numbered example scripts (01_*.py, 02_*.py, ..., NN_*.py).
    if f.endswith(".py") and len(f) >= 3 and f[0].isdigit() and f[1].isdigit():
        setattr(TestExamplesRun, f"test_{os.path.splitext(f)[0]}", _make_case(f))


if __name__ == "__main__":
    unittest.main()
