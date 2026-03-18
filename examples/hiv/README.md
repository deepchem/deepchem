# HIV Dataset Examples

The HIV dataset was introduced by the Drug Therapeutics
Program (DTP) AIDS Antiviral Screen, which tested the ability
to inhibit HIV replication for over 40,000 compounds.
Screening results were evaluated and placed into three
categories: confirmed inactive (CI),confirmed active (CA) and
confirmed moderately active (CM). We further combine the
latter two labels, making it a classification task between
inactive (CI) and active (CA and CM).

The data file contains a csv table, in which columns below
are used:
- "smiles": SMILES representation of the molecular structure
- "activity": Three-class labels for screening results: CI/CM/CA
- "HIV_active": Binary labels for screening results: 1 (CA/CM) and 0 (CI)

References:
AIDS Antiviral Screen Data. https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data

In this example we train models on the HIV collection.
