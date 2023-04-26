import hydra
from omegaconf import DictConfig
import deepchem as dc
from deepchem.models import GCNModel
from deepchem.models.lightning.dc_lightning_module import DCLightningModule
from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule, collate_dataset_wrapper
from deepchem.feat import MolGraphConvFeaturizer
import pytorch_lightning as pl


@hydra.main(version_base=None, config_path="configs", config_name="zinc15")
def main(cfg: DictConfig) -> None:
  featurizer = MolGraphConvFeaturizer()
  tasks, datasets, _ = dc.molnet.load_zinc15(featurizer=featurizer)
  _, valid_dataset, _ = datasets

  n_tasks = len(tasks)
  model = GCNModel(graph_conv_layers=cfg.graph_conv_layers,
                   mode=cfg.mode,
                   n_tasks=n_tasks,
                   number_atom_features=cfg.number_atom_features,
                   learning_rate=cfg.learning_rate)

  gcnmodule = DCLightningModule(model)
  smiles_datasetmodule = DCLightningDatasetModule(valid_dataset, cfg.batch_size,
                                                  collate_dataset_wrapper)

  trainer = pl.Trainer(
      max_epochs=cfg.max_epochs,
      devices=cfg.num_device,
      accelerator=cfg.device,
  )

  trainer.fit(gcnmodule, smiles_datasetmodule)


if __name__ == "__main__":
  main()
