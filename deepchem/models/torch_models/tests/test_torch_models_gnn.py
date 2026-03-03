def test_gatv2_forward_pass():
    import deepchem as dc
    from deepchem.models.torch_models import GATv2Model

    # Load sample dataset
    tasks, datasets, _ = dc.molnet.load_tox21()
    train = datasets[0]

    # Initialize model
    model = GATv2Model(n_tasks=len(tasks), mode='classification')

    # Test forward pass
    batch = train.X[0]  # Assume X contains PyG Data objects
    output = model.model(batch)
    assert output.shape == (len(batch), len(tasks))

def test_graphsage_overfit():
    from deepchem.models.torch_models import GraphSAGEModel

    # Create synthetic data
    dataset = ...  # Mock PyG dataset

    # Train model
    model = GraphSAGEModel(n_tasks=1), mode='regression')
    model.fit(dataset, nb_epoch=100)

    # Check loss decreases
    assert model.train_loss_history[-1] < 0.1 * model.train_loss_history[0]
