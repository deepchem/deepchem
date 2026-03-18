import os
import tempfile
import deepchem as dc


def test_load_singleton_csv():
    fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fin.write("smiles,endpoint\nc1ccccc1,1")
    fin.close()
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["endpoint"]
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    X = loader.create_dataset(fin.name)
    assert len(X) == 1
    os.remove(fin.name)
