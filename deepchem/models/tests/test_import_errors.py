import subprocess
import sys


def test_missing_tqdm_error_names_real_cause():
    code = "import sys; sys.modules['tqdm'] = None; import deepchem"
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True,
                            text=True)
    assert result.returncode != 0
    assert "tqdm" in result.stderr.splitlines()[-1]
