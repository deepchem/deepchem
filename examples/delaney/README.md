The Delaney dataset is a collection of 2874 aqueous solubility measurements from this paper:

Delaney, John S. "ESOL: estimating aqueous solubility directly from molecular structure." Journal of chemical information and computer sciences 44.3 (2004): 1000-1005.

This dataset is commonly used since it's a small molecular
regression dataset that's convenient for benchmarking various
techniques. In this example, we train a series of different
DeepChem models against this task:

- `DAGModel`: In `delaney_DAG.py`. This model will train and
converge very slowly.
- `TextCNNModel`: In `delaney_textcnn.py`. This model featurizes compounds as SMILES strings directly and trains a convolutional network directly on the text.
- `WeaveModel`: In `delaney_weave.py`. This model trains a weave style convolution on Delaney.
- `ChemCeption`: In `delaney_chemception.py`. This model trains a variant of an Inception convolutional network on images generated from molecules.
- `MPNNModel`: In `delaney_MPNN.py`. This model trains a little slower, but is faster than `DAGModel`.
