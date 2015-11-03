# Usage ./process_bace.sh INPUT_SDF_FILE
python -m deep_chem.scripts.process_dataset --input-file $1 --input-type sdf --fields Name smiles pIC50 --field-types string string concentration --name BACE --out /tmp/
