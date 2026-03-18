# README for Kinase Example

The Kinase dataset is an in-house dataset from Merck that was first introduced in the following paper:

Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

It contains 2500 Merck in-house compounds that were measured
for IC50 of inhibition on 99 protein kinases. Unlike most of
the other datasets featured in MoleculeNet, the Kinase
collection does not have structures for the compounds tested
since they were proprietary Merck compounds. However, the
collection does feature pre-computed descriptors for these
compounds.

Note that the original train/valid/test split from the source
data was preserved here, so this function doesn't allow for
alternate modes of splitting. Similarly, since the source data
came pre-featurized, it is not possible to apply alternative
featurizations.

This example features a few different models trained on this
dataset collection. In particular:

- `kinase_rf.py` trains a random forest model
