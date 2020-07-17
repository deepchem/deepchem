# SIDER Dataset Examples

The Side Effect Resource (SIDER) is a database of marketed
drugs and adverse drug reactions (ADR). The version of the
SIDER dataset in DeepChem has grouped drug side effects into
27 system organ classes following MedDRA classifications
measured for 1427 approved drugs.

The data file contains a csv table, in which columns below
are used:

- "smiles": SMILES representation of the molecular structure
- "Hepatobiliary disorders" ~ "Injury, poisoning and procedural complications": Recorded side effects for the drug

Please refer to http://sideeffects.embl.de/se/?page=98 for details on ADRs.

References:
Kuhn, Michael, et al. "The SIDER database of drugs and side effects." Nucleic acids research 44.D1 (2015): D1075-D1079.
Altae-Tran, Han, et al. "Low data drug discovery with one-shot learning." ACS central science 3.4 (2017): 283-293.
Medical Dictionary for Regulatory Activities. http://www.meddra.org/
