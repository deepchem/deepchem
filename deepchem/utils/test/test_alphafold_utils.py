import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from deepchem.utils import alphafold_utils


class MockResponse:

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
        return self.payload


def test_get_alphafold_structure_metadata_queries_api(monkeypatch):
    response = [{
        "entryId": "AF-P69905-F1",
        "uniprotAccession": "P69905",
        "uniprotId": "HBA_HUMAN",
        "pdbUrl": "https://example.org/AF-P69905-F1-model_v4.pdb",
        "cifUrl": "https://example.org/AF-P69905-F1-model_v4.cif",
        "bcifUrl": "https://example.org/AF-P69905-F1-model_v4.bcif",
        "paeDocUrl": "https://example.org/AF-P69905-F1-predicted_aligned_error_v4.json",
        "latestVersion": 4,
    }]
    opened_urls = []

    def mock_urlopen(request, timeout=10):
        opened_urls.append(request.full_url)
        return MockResponse(json.dumps(response))

    monkeypatch.setattr(alphafold_utils.request, "urlopen", mock_urlopen)

    metadata = alphafold_utils.get_alphafold_structure_metadata("p69905")

    assert opened_urls == [
        "https://alphafold.ebi.ac.uk/api/prediction/P69905"
    ]
    assert metadata.entry_id == "AF-P69905-F1"
    assert metadata.uniprot_accession == "P69905"
    assert metadata.uniprot_id == "HBA_HUMAN"
    assert metadata.pdb_url == "https://example.org/AF-P69905-F1-model_v4.pdb"
    assert metadata.version == 4
    assert metadata.license == "CC BY 4.0"


def test_get_alphafold_structure_metadata_rejects_empty_response(monkeypatch):

    def mock_urlopen(request, timeout=10):
        return MockResponse("[]")

    monkeypatch.setattr(alphafold_utils.request, "urlopen", mock_urlopen)

    with pytest.raises(ValueError, match="No AlphaFold DB prediction"):
        alphafold_utils.get_alphafold_structure_metadata("P00000")


def test_get_alphafold_structure_metadata_wraps_http_errors(monkeypatch):

    def mock_urlopen(request, timeout=10):
        raise HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(alphafold_utils.request, "urlopen", mock_urlopen)

    with pytest.raises(ValueError, match="No AlphaFold DB prediction"):
        alphafold_utils.get_alphafold_structure_metadata("P00000")


def test_download_alphafold_structure_writes_pdb(monkeypatch, tmp_path):
    metadata_response = [{
        "entryId": "AF-P69905-F1",
        "uniprotAccession": "P69905",
        "pdbUrl": "https://example.org/AF-P69905-F1-model_v4.pdb",
        "cifUrl": "https://example.org/AF-P69905-F1-model_v4.cif",
        "latestVersion": 4,
    }]
    pdb_text = "HEADER    ALPHAFOLD MODEL\nATOM      1  N   MET A   1\n"
    opened_urls = []

    def mock_urlopen(request, timeout=10):
        opened_urls.append(request.full_url)
        if request.full_url.endswith("/P69905"):
            return MockResponse(json.dumps(metadata_response))
        return MockResponse(pdb_text)

    monkeypatch.setattr(alphafold_utils.request, "urlopen", mock_urlopen)

    result = alphafold_utils.download_alphafold_structure("P69905", tmp_path)

    assert result.path == str(tmp_path / "AF-P69905-F1.pdb")
    assert Path(result.path).read_text() == pdb_text
    assert result.format == "pdb"
    assert result.metadata.entry_id == "AF-P69905-F1"
    assert opened_urls == [
        "https://alphafold.ebi.ac.uk/api/prediction/P69905",
        "https://example.org/AF-P69905-F1-model_v4.pdb",
    ]


def test_download_alphafold_structure_supports_mmcif(monkeypatch, tmp_path):
    metadata_response = [{
        "entryId": "AF-P69905-F1",
        "uniprotAccession": "P69905",
        "pdbUrl": "https://example.org/AF-P69905-F1-model_v4.pdb",
        "cifUrl": "https://example.org/AF-P69905-F1-model_v4.cif",
        "latestVersion": 4,
    }]

    def mock_urlopen(request, timeout=10):
        if request.full_url.endswith("/P69905"):
            return MockResponse(json.dumps(metadata_response))
        return MockResponse("data_AF-P69905-F1\n#\n")

    monkeypatch.setattr(alphafold_utils.request, "urlopen", mock_urlopen)

    result = alphafold_utils.download_alphafold_structure("P69905",
                                                          tmp_path,
                                                          file_format="mmcif")

    assert result.path == str(tmp_path / "AF-P69905-F1.cif")
    assert Path(result.path).read_text() == "data_AF-P69905-F1\n#\n"
    assert result.format == "mmcif"


def test_download_alphafold_structure_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="file_format must be"):
        alphafold_utils.download_alphafold_structure("P69905",
                                                     tmp_path,
                                                     file_format="pdbqt")
