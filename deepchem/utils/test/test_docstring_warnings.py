import unittest
import warnings
from pathlib import Path


class TestDocstringWarnings(unittest.TestCase):

    def test_affected_files_have_no_invalid_escape_warnings(self):
        repo_root = Path(__file__).resolve().parents[3]

        files = [
            "deepchem/models/torch_models/attention.py",
            "deepchem/models/torch_models/ferminet.py",
            "deepchem/models/torch_models/flows.py",
            "deepchem/models/torch_models/layers.py",
            "deepchem/utils/differentiation_utils/optimize/jacobian.py",
            "deepchem/utils/differentiation_utils/solve.py",
            "deepchem/utils/equivariance_utils.py",
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            for filename in files:
                path = repo_root / filename
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")
