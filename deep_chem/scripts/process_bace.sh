# Usage ./process_bace.sh INPUT_SDF_FILE OUT_DIR DATASET_NAME
python -m deep_chem.scripts.process_dataset --input-file $1 --input-type sdf --fields Name smiles pIC50 Model --field-types string string float string --name $3 --out $2 --prediction-endpoint pIC50
