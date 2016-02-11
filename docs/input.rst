.. _input:

Input Format
============
Accepted input formats include csv, pkl.gz, and sdf file. For simplicity,
let's assume we deal with a csv input. In order to build models, we expect
the following columns to have entries for each row in the csv file.

1. A column containing SMILES strings [1].
2. A column containing an experimental measurement.
3. (Optional) A column containing a unique compound identifier.

Here's an example of a potential input file. 

+---------------+-------------------------------------------+----------------+ 
|Compound ID    | measured log solubility in mols per litre | smiles         | 
+===============+===========================================+================+ 
| benzothiazole | -1.5                                      | c2ccc1scnc1c2  | 
+---------------+-------------------------------------------+----------------+ 

Here the "smiles" column contains the SMILES string, the "measured log
solubility in mols per litre" contains the experimental measurement and
"Compound ID" contains the unique compound identifier.

[2] Anderson, Eric, Gilman D. Veith, and David Weininger. "SMILES, a line
notation and computerized interpreter for chemical structures." US
Environmental Protection Agency, Environmental Research Laboratory, 1987.

