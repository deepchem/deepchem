import pytest
import tempfile

import deepchem as dc
import numpy as np

try:
    import torch
    from deepchem.models.torch_models.scscore import SCScoreModel
    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_scscore_reload():
    """Tests the model reload capability"""

    #create a random dataset
    n_samples = 100
    n_features = 32
    X = np.random.rand(n_samples, 2, n_features) #2 indicates we need two molecular features for each sample
    dataset = dc.data.NumpyDataset(X) #SCScore don't use labels

    model_dir = tempfile.mkdtemp() #create an empty directory for saving the model

    #initialize the model
    model = SCScoreModel(n_features=n_features, model_dir=model_dir)
    model.fit(dataset, nb_epoch=25)
    model_preds = model.predict(dataset)

    #reload the model
    reloaded_model = SCScoreModel(n_features=n_features, model_dir=model_dir)
    reloaded_model.restore()
    reloaded_model_preds = reloaded_model.predict(dataset)

    assert len(model_preds) == len(reloaded_model_preds)
    for i,j in zip(model_preds, reloaded_model_preds):
      assert np.all(i == j), "SCScoreModel failed in the model reload test"
  

@pytest.mark.torch
def test_scscore_overfit():
    """Tests the model for overfit"""
    
    #create a random dataset
    n_samples = 100
    n_features = 32
    X = np.random.rand(n_samples, 2, n_features) #2 indicates we need two molecular features for each sample
    dataset = dc.data.NumpyDataset(X) #SCScore don't use labels

    #initialize the model
    model = SCScoreModel(n_features=n_features)
    model.fit(dataset, nb_epoch=25)
    preds = model.predict(dataset)

    assert np.array_equal(np.zeros(shape=preds[0].shape), preds[0] > preds[1]), "SCScoreModel failed in the overfitting test"

if __name__ == "__main__":
    test_scscore_reload()
    test_scscore_overfit()
