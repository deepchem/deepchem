"""Utilities for retrieving AlphaFold Protein Structure Database models.

The AlphaFold Protein Structure Database provides precomputed protein
structure predictions for UniProt accessions. These helpers intentionally do
not depend on the original AlphaFold, OpenFold, ColabFold, JAX, PyTorch, model
weights, or sequence databases. They provide a lightweight integration point so
DeepChem workflows can retrieve predicted structures and pass the resulting PDB
or mmCIF files to existing featurizers and docking tools.

Examples
--------
>>> from deepchem.utils.alphafold_utils import download_alphafold_structure
>>> # Downloads AF-P69905-F1.pdb into the current directory.
>>> # result = download_alphafold_structure("P69905", ".")
>>> # result.path
>>> # './AF-P69905-F1.pdb'
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

ALPHAFOLD_DB_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
ALPHAFOLD_DB_LICENSE = "CC BY 4.0"


@dataclass(frozen=True)
class AlphaFoldStructureMetadata:
    """Metadata for a prediction in the AlphaFold Protein Structure Database.

    Parameters
    ----------
    entry_id: str
        AlphaFold DB entry identifier, for example ``"AF-P69905-F1"``.
    uniprot_accession: str
        UniProt accession used by the AlphaFold DB prediction.
    uniprot_id: str, optional
        UniProt mnemonic identifier if returned by the API.
    pdb_url: str, optional
        URL for the predicted structure in PDB format.
    cif_url: str, optional
        URL for the predicted structure in mmCIF format.
    bcif_url: str, optional
        URL for the predicted structure in binary CIF format.
    pae_doc_url: str, optional
        URL for predicted aligned error JSON metadata.
    version: int, optional
        Latest AlphaFold DB model version.
    license: str
        License for AlphaFold DB structures. AlphaFold DB structures are
        provided under CC BY 4.0.
    raw_metadata: dict, optional
        Unmodified JSON object returned by the AlphaFold DB API.
    """
    entry_id: str
    uniprot_accession: str
    uniprot_id: Optional[str] = None
    pdb_url: Optional[str] = None
    cif_url: Optional[str] = None
    bcif_url: Optional[str] = None
    pae_doc_url: Optional[str] = None
    version: Optional[int] = None
    license: str = ALPHAFOLD_DB_LICENSE
    raw_metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class AlphaFoldStructure:
    """A downloaded AlphaFold DB structure file and its metadata.

    Parameters
    ----------
    path: str
        Path to the downloaded structure file.
    format: str
        File format written to ``path``. Supported values are ``"pdb"`` and
        ``"mmcif"``.
    metadata: AlphaFoldStructureMetadata
        Metadata returned by the AlphaFold DB API.
    """
    path: str
    format: str
    metadata: AlphaFoldStructureMetadata


def _normalize_uniprot_id(uniprot_id: str) -> str:
    """Normalize and validate a UniProt accession for AlphaFold DB queries."""
    normalized = uniprot_id.strip().upper()
    if not normalized:
        raise ValueError("uniprot_id must be a non-empty string")
    return normalized


def _read_url(url: str, timeout: float) -> bytes:
    """Read bytes from a URL using only the Python standard library."""
    req = request.Request(url, headers={"User-Agent": "deepchem-alphafold-utils"})
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _metadata_from_api_record(record: Dict[str, Any]) -> AlphaFoldStructureMetadata:
    """Convert an AlphaFold DB API record into a metadata dataclass."""
    entry_id = record.get("entryId")
    uniprot_accession = record.get("uniprotAccession")
    if not entry_id or not uniprot_accession:
        raise ValueError("AlphaFold DB response is missing required metadata")

    return AlphaFoldStructureMetadata(
        entry_id=entry_id,
        uniprot_accession=uniprot_accession,
        uniprot_id=record.get("uniprotId"),
        pdb_url=record.get("pdbUrl"),
        cif_url=record.get("cifUrl"),
        bcif_url=record.get("bcifUrl"),
        pae_doc_url=record.get("paeDocUrl"),
        version=record.get("latestVersion"),
        raw_metadata=record,
    )


def get_alphafold_structure_metadata(
        uniprot_id: str,
        timeout: float = 10.0) -> AlphaFoldStructureMetadata:
    """Get metadata for an AlphaFold DB prediction by UniProt accession.

    Parameters
    ----------
    uniprot_id: str
        UniProt accession, for example ``"P69905"``.
    timeout: float, default 10.0
        Network timeout in seconds.

    Returns
    -------
    AlphaFoldStructureMetadata
        Metadata for the first AlphaFold DB prediction matching ``uniprot_id``.

    Raises
    ------
    ValueError
        If no AlphaFold DB prediction exists for the accession, the API returns
        malformed data, or the request fails.
    """
    normalized_uniprot_id = _normalize_uniprot_id(uniprot_id)
    url = ALPHAFOLD_DB_API_URL.format(uniprot_id=normalized_uniprot_id)

    try:
        payload = _read_url(url, timeout)
    except error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                f"No AlphaFold DB prediction found for {normalized_uniprot_id}"
            ) from exc
        raise ValueError(
            f"Failed to query AlphaFold DB for {normalized_uniprot_id}: {exc}"
        ) from exc
    except error.URLError as exc:
        raise ValueError(
            f"Failed to query AlphaFold DB for {normalized_uniprot_id}: {exc}"
        ) from exc

    try:
        records = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("AlphaFold DB returned invalid JSON") from exc

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(
            f"No AlphaFold DB prediction found for {normalized_uniprot_id}")
    if not isinstance(records[0], dict):
        raise ValueError("AlphaFold DB response has an unexpected format")

    return _metadata_from_api_record(records[0])


def download_alphafold_structure(uniprot_id: str,
                                 output_dir: str,
                                 file_format: str = "pdb",
                                 timeout: float = 10.0,
                                 filename: Optional[str] = None
                                 ) -> AlphaFoldStructure:
    """Download a predicted AlphaFold DB structure by UniProt accession.

    Parameters
    ----------
    uniprot_id: str
        UniProt accession, for example ``"P69905"``.
    output_dir: str
        Directory where the structure file should be written. The directory is
        created if it does not already exist.
    file_format: str, default "pdb"
        Structure format to download. Supported values are ``"pdb"`` and
        ``"mmcif"``. ``"cif"`` is accepted as an alias for ``"mmcif"``.
    timeout: float, default 10.0
        Network timeout in seconds for both metadata and file download.
    filename: str, optional
        Output filename. If omitted, a filename derived from the AlphaFold DB
        entry identifier is used.

    Returns
    -------
    AlphaFoldStructure
        Path, format, and metadata for the downloaded structure.

    Raises
    ------
    ValueError
        If ``file_format`` is unsupported, metadata is unavailable, the selected
        format is unavailable for the entry, or a network request fails.
    """
    normalized_format = file_format.lower()
    if normalized_format == "cif":
        normalized_format = "mmcif"
    if normalized_format not in {"pdb", "mmcif"}:
        raise ValueError("file_format must be one of 'pdb', 'mmcif', or 'cif'")

    metadata = get_alphafold_structure_metadata(uniprot_id, timeout=timeout)
    structure_url = metadata.pdb_url if normalized_format == "pdb" else metadata.cif_url
    if structure_url is None:
        raise ValueError(
            f"AlphaFold DB entry {metadata.entry_id} does not provide {normalized_format} data"
        )

    suffix = ".pdb" if normalized_format == "pdb" else ".cif"
    if filename is None:
        filename = f"{metadata.entry_id}{suffix}"

    output_path = Path(output_dir) / filename
    os.makedirs(output_path.parent, exist_ok=True)

    try:
        structure_bytes = _read_url(structure_url, timeout)
    except error.URLError as exc:
        raise ValueError(
            f"Failed to download AlphaFold DB structure {metadata.entry_id}: {exc}"
        ) from exc

    output_path.write_bytes(structure_bytes)
    return AlphaFoldStructure(path=str(output_path),
                              format=normalized_format,
                              metadata=metadata)
