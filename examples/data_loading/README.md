# Data Loading Examples

The examples in this directory highlight a number of ways to
load datasets into DeepChem for downstream analysis:

- `pandas_csv.py` shows how to directly load a dataset from a CSV file without using a `DataLoader`.
- `sdf_load.py` shows how to load a dataset from a sdf file using `SDFLoader`.

## Notes on Optional Dependencies and Warnings

When running the examples in this directory, you may see warning messages such as:

- Messages about missing optional dependencies (e.g. PyTorch, TensorFlow, JAX)
- Notices that certain models or utilities were skipped
- Deprecation warnings related to featurizers (e.g. Morgan fingerprints)

These warnings are expected and do **not** indicate a failure.

### Optional Machine Learning Backends

DeepChem supports multiple optional machine learning backends, including PyTorch, TensorFlow, and JAX.  
The data loading examples in this directory do **not** require these libraries to be installed.  
If a backend is not available, DeepChem will skip loading the corresponding modules and continue normally.

### RDKit Dependency

Some examples (such as `sdf_load.py`) require RDKit to be installed for molecular featurization.  
If RDKit is not available, these examples will fail.

### Deprecation Warnings

Some examples currently use APIs that are being deprecated (for example, older featurization interfaces).  
These warnings are informational and indicate planned changes in future DeepChem releases.
