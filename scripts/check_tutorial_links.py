#!/usr/bin/env python
"""Check that every reference to a tutorial notebook points at a real file.

Three places in the repository name a tutorial notebook, and all three rot
silently when a notebook is renamed:

1. the Colab badge and cross-references carried inside the notebooks,
2. the index in ``examples/tutorials/README.md``,
3. ``examples/tutorials/website-render-order/*.csv``, which drives the website.

Renaming a notebook has broken these before, so this script reports them
instead of waiting for a reader to hit a 404. Links pinned to a ref other than
the one being checked, and links that leave the repository, are ignored: they
may legitimately name a file that is absent here.

Usage
-----
::

    python scripts/check_tutorial_links.py [ref]

``ref`` defaults to ``master``. The script exits with status 1 if any
reference is broken.
"""
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterator, List, NamedTuple, Set, Tuple

# Every deepchem URL that names a notebook has this shape, whether it is a
# Colab badge (served from colab.research.google.com) or a plain GitHub link.
_URL = re.compile(r"https://\S*?/deepchem/deepchem/blob/([^/]+)/(\S+?\.ipynb)")
_COLAB = re.compile(r"colab\.research\.google\.com")
_MD_LINK = re.compile(r"\]\(([^)\s]+\.ipynb)\)")


class BrokenLink(NamedTuple):
    """A reference to a tutorial notebook that does not exist.

    Attributes
    ----------
    kind: str
        What sort of reference this is, for grouping in the report.
    source: str
        Repository-relative path of the file containing the reference.
    detail: str
        Cell index or other locator within `source`, if any.
    target: str
        The notebook the reference names, which could not be found.
    """

    kind: str
    source: str
    detail: str
    target: str


def _iter_notebook_urls(nb_path: Path,
                        ref: str) -> Iterator[Tuple[int, str, bool]]:
    """Yield the notebook URLs found in every cell of `nb_path`.

    Parameters
    ----------
    nb_path: Path
        Notebook to read.
    ref: str
        Only URLs pinned to this git ref are returned.

    Yields
    ------
    Tuple[int, str, bool]
        A `(cell_index, path, is_colab)` triple, where `path` is the
        repository-relative path the URL names.
    """
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        for match in _URL.finditer(source):
            url_ref, path = match.group(1), match.group(2)
            if url_ref == ref:
                yield index, path, bool(_COLAB.search(match.group(0)))


def _check_notebooks(tutorials: Path, repo: Path, ref: str) -> List[BrokenLink]:
    """Check the Colab badges and cross-references inside the notebooks.

    Parameters
    ----------
    tutorials: Path
        The `examples/tutorials` directory.
    repo: Path
        Repository root, against which link paths are resolved.
    ref: str
        Only URLs pinned to this git ref are checked.

    Returns
    -------
    List[BrokenLink]
        One entry per reference whose target is missing.
    """
    broken = []
    for nb_path in sorted(tutorials.glob("*.ipynb")):
        for index, path, is_colab in _iter_notebook_urls(nb_path, ref):
            if not (repo / path).exists():
                broken.append(
                    BrokenLink("Colab badge" if is_colab else "cross-reference",
                               f"examples/tutorials/{nb_path.name}",
                               f"cell {index}",
                               path.rsplit("/", 1)[-1]))
    return broken


def _check_readme(tutorials: Path, ref: str) -> List[BrokenLink]:
    """Check the notebook links in the tutorial index.

    Parameters
    ----------
    tutorials: Path
        The `examples/tutorials` directory.
    ref: str
        Only URLs pinned to this git ref are checked.

    Returns
    -------
    List[BrokenLink]
        One entry per index link whose target is missing.
    """
    broken = []
    readme = tutorials / "README.md"
    for target in _MD_LINK.findall(readme.read_text(encoding="utf-8")):
        if "://" in target:
            match = _URL.search(target)
            if match is None or match.group(1) != ref:
                continue  # another repository, or pinned to another ref
            path = match.group(2)
        else:
            path = target
        if not (tutorials / path).exists():
            broken.append(
                BrokenLink("README index", "examples/tutorials/README.md", "",
                           path.rsplit("/", 1)[-1]))
    return broken


def _check_render_order(tutorials: Path, present: Set[str]) -> List[BrokenLink]:
    """Check the file names listed in the website render-order CSVs.

    Parameters
    ----------
    tutorials: Path
        The `examples/tutorials` directory.
    present: Set[str]
        Names of the notebooks that exist in this checkout.

    Returns
    -------
    List[BrokenLink]
        One entry per listed notebook that is missing.
    """
    broken = []
    for csv_path in sorted((tutorials / "website-render-order").glob("*.csv")):
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("File Name") or "").strip()
                if name and "://" not in name and name not in present:
                    broken.append(
                        BrokenLink(
                            "render-order CSV",
                            "examples/tutorials/website-render-order/"
                            f"{csv_path.name}", "", name))
    return broken


def main() -> int:
    """Report every broken tutorial-notebook reference in this checkout.

    Returns
    -------
    int
        ``0`` when every reference resolves, ``1`` when any is broken, and
        ``2`` when no tutorial directory could be found.
    """
    repo = Path(__file__).resolve().parent.parent
    tutorials = repo / "examples" / "tutorials"
    if not tutorials.is_dir():
        print(f"error: {tutorials} not found; run this from a source checkout",
              file=sys.stderr)
        return 2

    ref = sys.argv[1] if len(sys.argv) > 1 else "master"
    present = {p.name for p in tutorials.glob("*.ipynb")}

    broken = (_check_notebooks(tutorials, repo, ref) +
              _check_readme(tutorials, ref) +
              _check_render_order(tutorials, present))

    if not broken:
        print(f"all references to {len(present)} tutorial notebooks resolve "
              f"(ref={ref})")
        return 0

    print(f"{len(broken)} broken reference(s) to tutorial notebooks "
          f"(ref={ref}):\n")
    for link in broken:
        print(f"[{link.kind}] {link.source} {link.detail}".rstrip())
        print(f"    -> {link.target}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
