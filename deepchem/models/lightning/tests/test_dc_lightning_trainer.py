import pytest
import torch
import deepchem as dc
try:
    import lightning as L
    from deepchem.models.lightning.trainer2 import DeepChemLightningTrainer
except ImportError as e:
    print(f"DeepChem Lightning module not found: {e}")
    pytest.skip("DeepChem Lightning module not found, skipping tests.",
                allow_module_level=True)


@pytest.mark.torch
def test_multitask_classifier():
    L.seed_everything(42)
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets

    model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                          n_features=1024,
                                          layer_sizes=[1000],
                                          dropouts=0.2,
                                          learning_rate=0.0001,
                                          device="cpu",
                                          batch_size=16)

    trainer = DeepChemLightningTrainer(model=model,
                                       batch_size=16,
                                       max_epochs=30,
                                       accelerator="cuda",
                                       devices=-1,
                                       log_every_n_steps=1,
                                       strategy="fsdp",
                                       fast_dev_run=True)

    trainer.fit(valid_dataset)
    # get a some 10 weights for assertion
    weights = trainer.model.model.layers[0].weight[:10].detach().cpu().numpy()

    trainer.save_checkpoint("multitask_classifier.ckpt")

    # Reload model and checkpoint
    model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                          n_features=1024,
                                          layer_sizes=[1000],
                                          dropouts=0.2,
                                          learning_rate=0.0001,
                                          device="cpu",
                                          batch_size=16)

    trainer = DeepChemLightningTrainer.load_checkpoint(
        "multitask_classifier.ckpt",
        model=model,
        batch_size=16,
        max_epochs=10,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
        strategy="fsdp",
        fast_dev_run=True)

    # get a some 10 weights for assertion
    reloaded_weights = trainer.model.model.layers[0].weight[0][:10].detach(
    ).cpu().numpy()

    _ = trainer.predict(valid_dataset)

    # make it equal with a tolerance of 1e-5
    assert torch.allclose(torch.tensor(weights),
                          torch.tensor(reloaded_weights),
                          atol=1e-5)


@pytest.mark.torch
def test_gcn_model():
    L.seed_everything(42)
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks, all_dataset, transformers = dc.molnet.load_bace_classification(
        featurizer)
    train_dataset, valid_dataset, test_dataset = all_dataset

    model = dc.models.GCNModel(mode='classification',
                               n_tasks=len(tasks),
                               batch_size=16,
                               learning_rate=0.001,
                               device="cpu")

    trainer = DeepChemLightningTrainer(model=model,
                                       batch_size=16,
                                       max_epochs=10,
                                       accelerator="cuda",
                                       devices=-1,
                                       log_every_n_steps=1,
                                       strategy="fsdp",
                                       fast_dev_run=True)

    trainer.fit(valid_dataset)

    # get a some 10 weights for assertion
    weights = trainer.model.model.model.gnn.gnn_layers[
        0].res_connection.weight[:10].detach().cpu().numpy()

    trainer.save_checkpoint("gcn_model.ckpt")

    # Reload model and checkpoint
    model = dc.models.GCNModel(mode='classification',
                               n_tasks=len(tasks),
                               batch_size=16,
                               learning_rate=0.001,
                               device="cpu")

    trainer = DeepChemLightningTrainer.load_checkpoint("gcn_model.ckpt",
                                                       model=model,
                                                       batch_size=16,
                                                       max_epochs=10,
                                                       accelerator="cuda",
                                                       devices=-1,
                                                       log_every_n_steps=1,
                                                       strategy="fsdp",
                                                       fast_dev_run=True)

    # get a some 10 weights for assertion
    reloaded_weights = trainer.model.model.model.gnn.gnn_layers[
        0].res_connection.weight[0][:10].detach().cpu().numpy()

    _ = trainer.predict(valid_dataset)

    # make it equal with a tolerance of 1e-5
    assert torch.allclose(torch.tensor(weights),
                          torch.tensor(reloaded_weights),
                          atol=1e-5)
