import deepchem as dc
import numpy as np

def test_dtnn_model_initialization():
    """Test DTNN model can be initialized"""
    model = dc.models.DTNNModel(
        n_tasks=1,
        n_embedding=30,
        n_hidden=100,
        n_distance=100
    )
    assert model is not None

def test_qm9_data_loading():
    """Verify QM9 dataset loads correctly"""
    tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='CoulombMatrix')
    train, valid, test = datasets
    
    assert len(train) > 0
    assert len(test) > 0
    assert train.X.shape[1] > 0  # Features exist

def test_training_improves_loss():
    """Ensure training actually reduces loss"""
    # Load small subset for quick test
    tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='CoulombMatrix')
    train, _, _ = datasets
    
    model = dc.models.DTNNModel(n_tasks=len(tasks))
    
    initial_loss = model.evaluate(train, metrics=[dc.metrics.Metric(dc.metrics.mae_score)])
    model.fit(train, nb_epoch=5)
    final_loss = model.evaluate(train, metrics=[dc.metrics.Metric(dc.metrics.mae_score)])
    
    assert final_loss['mae_score'] < initial_loss['mae_score']
