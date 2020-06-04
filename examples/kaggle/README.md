# Kaggle Dataset Examples

The Kaggle dataset is an in-house dataset from Merck that was first introduced in the following paper:

Ma, Junshui, et al. "Deep neural nets as a method for quantitative structure–activity relationships." Journal of chemical information and modeling 55.2 (2015): 263-274.

It contains 100,000 unique Merck in-house compounds that were
measured on 15 enzyme inhibition and ADME/TOX datasets.
Unlike most of the other datasets featured in MoleculeNet,
the Kaggle collection does not have structures for the
compounds tested since they were proprietary Merck compounds.
However, the collection does feature pre-computed descriptors
for these compounds.

Note that the original train/valid/test split from the source
data was preserved here, so this function doesn't allow for
alternate modes of splitting. Similarly, since the source data
came pre-featurized, it is not possible to apply alternative
featurizations.

This folder contains examples training models on the Kaggle dataset.
