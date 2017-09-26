import os
import subprocess
import tempfile

import nbformat


def _notebook_read(path):
  """
  Parameters
  ----------
  path: str
  path to ipython notebook

  Returns
  -------
  nb: notebook object
  errors: list of Exceptions
  """

  with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
    args = [
        "jupyter-nbconvert", "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=600", "--output", fout.name, path
    ]
    subprocess.check_call(args)

    fout.seek(0)
    nb = nbformat.read(fout, nbformat.current_nbformat)

  errors = [output for cell in nb.cells if "outputs" in cell
            for output in cell["outputs"] \
            if output.output_type == "error"]

  return nb, errors


def test_protein_ligand_complex_notebook():
  nb, errors = _notebook_read("protein_ligand_complex_notebook.ipynb")
  assert errors == []


def test_bace():
  nb, errors = _notebook_read("BACE.ipynb")
  assert errors == []


def test_multitask_networks_on_muv():
  nb, errors = _notebook_read("Multitask_Networks_on_MUV.ipynb")
  assert errors == []


def test_mnist():
  nb, errors = _notebook_read("mnist.ipynb")
  assert errors == []


def test_solubility():
  nb, errors = _notebook_read("solubility.ipynb")
  assert errors == []


def test_quantum():
  nb, errors = _notebook_read("quantum_machine_gdb1k.ipynb")
  assert errors == []


def test_pong():
  nb, errors = _notebook_read("pong.ipynb")
  assert errors == []


def test_graph_conv():
  nb, errors = _notebook_read("graph_convolutional_networks_for_tox21.ipynb")
  assert errors == []


def test_tg_mechanics():
  nb, errors = _notebook_read("TensorGraph_Mechanics.ipynb")
  assert errors == []


def test_seqtoseq_fingerprint():
  nb, errors = _notebook_read("seqtoseq_fingerprint.ipynb")
  assert errors == []


def test_dataset_preparation():
  nb, errors = _notebook_read("dataset_preparation.ipynb")
  assert errors == []
