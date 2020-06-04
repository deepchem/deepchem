# UV Examples

The UV dataset is an in-house dataset from Merck that was first introduced in the following paper:

Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

The UV dataset tests 10,000 of Merck's internal compounds on
190 absorption wavelengths between 210 and 400 nm. Unlike
most of the other datasets featured in MoleculeNet, the UV
collection does not have structures for the compounds tested
since they were proprietary Merck compounds. However, the
collection does feature pre-computed descriptors for these
compounds.

Note that the original train/valid/test split from the source
data was preserved here, so this function doesn't allow for
alternate modes of splitting. Similarly, since the source data
came pre-featurized, it is not possible to apply alternative
featurizations.
