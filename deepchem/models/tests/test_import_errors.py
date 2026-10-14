import subprocess
import sys

import pytest


@pytest.mark.torch
def test_missing_non_torch_dependency_is_reported():
    """If a non-PyTorch dependency of the torch layers is missing (here tqdm,
    which is imported via flows.py), the ImportError should name it rather
    than only claiming that PyTorch is not installed."""
    code = "import sys; sys.modules['tqdm'] = None; import deepchem"
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True,
                            text=True)
    assert result.returncode != 0
    assert 'PyTorch to be installed. Original error' in result.stderr
    assert 'tqdm' in result.stderr
    # The original exception should be chained with ``raise ... from e``.
    assert 'direct cause' in result.stderr
