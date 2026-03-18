# mol2vec implementation

In the recent mol2vec [paper](https://chemrxiv.org/articles/Mol2vec_Unsupervised_Machine_Learning_Approach_with_Chemical_Intuition/5513581), authors Jaeger et al consider the features returned by the rdkit Morgan fingerprint as "words" and a compound as a "sentence" to generate fixed-length embeddings. In this case we reproduce 200-element embeddings via a download of all SDF files in the PubChem compound database

## Setup

Ensure that gensim is installed via:

```bash
pip install gensim
```

## Creating training corpus

First, download the pubchem compound SDF corpus via running:

```bash
python ../pubchem_dataset/download_pubchem_ftp.sh
```
Note - the script assumes that a /media/data/pubchem directory exists for this large download (approx 19 GB as of November 2017)

Then generate the embeddings file via:

```bash
./train_mol2vec.sh
```

Then you can use these embeddings as a fixed-length alternative to fingerprints derived directly from RDKit. A full implementation as a featurized for deepchem is WIP

Example code for using the vec.txt file that is created by the above script can be found in eval_mol2vec_results