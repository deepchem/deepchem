import deepchem as dc
from deepchem.models import GCNModel
from deepchem.models.lightning.dc_lightning_module import DCLightningModule
from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule, collate_dataset_wrapper
from deepchem.feat import MolGraphConvFeaturizer
import pytorch_lightning as pl
import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--device",
                      type=str,
                      required=True,
                      choices=["cpu", 'gpu'])
  args = parser.parse_args()
  featurizer = MolGraphConvFeaturizer()
  tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer=featurizer)
  _, valid_dataset, test_dataset = datasets

  n_tasks = len(tasks)
  model = GCNModel(graph_conv_layers=[1024, 1024, 1024, 512, 512, 512],
                   mode='regression',
                   n_tasks=n_tasks,
                   number_atom_features=30,
                   batch_size=1024,
                   learning_rate=0.001)

  gcnmodule = DCLightningModule(model)
  smiles_datasetmodule = DCLightningDatasetModule(valid_dataset, 1024,
                                                  collate_dataset_wrapper)

  trainer = pl.Trainer(
      max_epochs=1,
      profiler="simple",
      devices=1,
      accelerator=args.device,
  )

  trainer.fit(gcnmodule, smiles_datasetmodule)


if __name__ == "__main__":
  main()
