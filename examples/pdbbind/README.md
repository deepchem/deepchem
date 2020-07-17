This example trains models on the PDBBind Dataset:

Wang, Renxiao, et al. "The PDBbind database: methodologies and updates." Journal of medicinal chemistry 48.12 (2005): 4111-4119.

In particular, on the 2015 release of PDBBind. There are three
subsets of this dataset, "core", "refined", and "full." Core is
about a couple hundred structures, refined a couple thousand,
and "full" about 10 thousand.

You can load a version of this dataset by calling
`dc.molnet.load_pdbbind()`. Make sure to have environment
variable `DEEPCHEM_DATA_DIR` set to point somewhere meaningful
otherwise you'll download the full PDBBind dataset every time.
You can use the "grid" or "atomic" featurizations for this
datasets. We'll provide examples of both.

Note that computing the atomic featurizations is slow, so we recommend doing simple examples on "core" to avoid experiments taking too long.


- `dc.models.AtomicConvModel`: Trained in `pdbbind_atomic_conv.py`
