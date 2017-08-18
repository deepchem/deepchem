import os
import sys
import subprocess
import tempfile

from nose.tools import nottest


def test_adme():
  sys.path.append(0, './adme')
  import run_benchmarks
  run_benchmarks.main()
  del run_benchmarks
  sys.path.remove('./adme')


if __name__ == "__main__":
  test_adme()
