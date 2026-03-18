# Toxcast Examples

ToxCast is an extended data collection from the same
initiative as Tox21, providing toxicology data for a large
library of compounds based on in vitro high-throughput
screening. The processed collection includes qualitative
results of over 600 experiments on 8k compounds.

The source data file contains a csv table, in which columns
below are used:

- "smiles": SMILES representation of the molecular structure
- "ACEA_T47D_80hr_Negative" ~ "Tanguay_ZF_120hpf_YSE_up": Bioassays results. Please refer to the section "high-throughput assay information" at https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data for details.

The source paper is 

Richard, Ann M., et al. "ToxCast chemical landscape: paving the road to 21st century toxicology." Chemical research in toxicology 29.8 (2016): 1225-1251.

In this example, we train a Random Forest model on the Toxcast dataset.
