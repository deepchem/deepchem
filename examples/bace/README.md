# BACE Dataset Examples

The BACE dataset is from the following paper:

Subramanian, Govindan, et al. "Computational modeling of β-secretase 1 (BACE-1) inhibitors using ligand based approaches." Journal of chemical information and modeling 56.10 (2016): 1936-1949.

This study considers a small dataset of 205 compounds datasets
which are used to train a model which is evaluated on a larger
external validation set of 1273 compounds.

The file `bace_datasets.py` loads the data as used in the
original paper. `bace_rf.py` demonstrates training a random
forest against this dataset.
