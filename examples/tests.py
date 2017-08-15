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
  return subprocess.check_output(cmd)


def test_adme():
  result, output = _example_run("./adme/run_benchmarks.py")
  print(output)
  assert result == 0


if __name__ == "__main__":
  test_adme()
