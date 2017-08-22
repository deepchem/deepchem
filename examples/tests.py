import os
import subprocess
import tempfile


def _example_run(path):
  """Checks that the example at the specified path runs corrently.

  Parameters
  ----------
  path: str
    Path to example file.
  Returns
  -------
  result: int 
    Return code. 0 for success, failure otherwise.
  """
  cmd = ["python", path]
  # Will raise a CalledProcessError if fails.
  retval = subprocess.check_output(cmd)
  return retval


def test_adme():
  print("Running test_adme()")
  output = _example_run("./adme/run_benchmarks.py")
  print(output)


def test_tox21_fcnet():

  print("Running tox21_fcnet()")
  output = _example_run("./tox21/tox21_fcnet.py")
  print(output)


if __name__ == "__main__":
  test_tox21_fcnet()
  test_adme()
