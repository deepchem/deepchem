from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

from new_dc_lightning_dataset_module import DCLightningDataModule
from new_dc_lightning_module import DeepChemLightningModule
import deepchem as dc
import lightning as L

class DeepChemLightningTrainer:
    """
    A wrapper class that handles the training and inference of DeepChem models using Lightning.
    
    Args:
        model: Initialized DeepChem model
        batch_size: Batch size for training
        **trainer_kwargs: Keyword arguments for Lightning Trainer
    """
    def __init__(
        self,
        model,
        batch_size: int = 32,
        **trainer_kwargs
    ):
        self.model = model
        self.batch_size = batch_size
        self.trainer_kwargs = trainer_kwargs
        
        # Set default trainer arguments if not provided
        if 'accelerator' not in trainer_kwargs:
            self.trainer_kwargs['accelerator'] = 'auto'
        if 'devices' not in trainer_kwargs:
            self.trainer_kwargs['devices'] = 'auto'
        if 'strategy' not in trainer_kwargs:
            self.trainer_kwargs['strategy'] = 'auto'
        if 'max_epochs' not in trainer_kwargs:
            self.trainer_kwargs['max_epochs'] = 100
            
        # Set up logger if not provided
        # if 'logger' not in trainer_kwargs:
        #     self.trainer_kwargs['logger'] = TensorBoardLogger("tb_logs", name="deepchem")
            
        # Here, we can use any custom logger used in deepchem
        # if 'callbacks' not in trainer_kwargs:
        #     self.trainer_kwargs['callbacks'] = [CustomProgressBar()]
            
        # Create the Lightning module
        self.lightning_model = DeepChemLightningModule(model)


        
    def fit(
        self,
        train_dataset: dc.data.Dataset,
        num_workers: int = 4
    ):
        """
        Train the model on the provided dataset.
        
        Args:
            train_dataset: DeepChem dataset for training
            num_workers: Number of workers for DataLoader
            
        Returns:
            The trainer object after fitting
        """
        # Set log_every_n_steps if not provided
        if 'log_every_n_steps' not in self.trainer_kwargs:
            dataset_size = len(train_dataset)
            self.trainer_kwargs['log_every_n_steps'] = max(1, dataset_size // (self.batch_size * 2))
        
        # Create data module
        data_module = DCLightningDataModule(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            model=self.model
        )
        
        # Create trainer
        self.trainer = L.Trainer(**self.trainer_kwargs)
        
        # Train the model
        self.trainer.fit(self.lightning_model, data_module)
        
    
    def predict(
        self,
        dataset: dc.data.Dataset,
        transformers  = [],
        other_output_types = None,
        num_workers: int = 4,
    ):
        """
        Run inference on the provided dataset.
        
        Args:
            dataset: DeepChem dataset for prediction
            num_workers: Number of workers for DataLoader
            uncertainty: Whether to compute uncertainty estimates
            transformers: List of transformers to apply to predictions
            other_output_types: List of other output types to compute
            
        Returns:
            Predictions from the model
        """
        if not hasattr(self, 'trainer'):
            self.trainer = L.Trainer(**self.trainer_kwargs)
            
        # Create data module
        data_module = DCLightningDataModule(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            model=self.model
        )
        self.lightning_model.eval()
        # Run prediction

        self.lightning_model.transformers = transformers
        self.lightning_model.other_output_types = other_output_types

        predictions = self.trainer.predict(
            self.lightning_model,
            data_module,
            return_predictions=True
        )
        
        return predictions
    
    def save_checkpoint(self, filepath: str):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        if hasattr(self, 'trainer'):
            self.trainer.save_checkpoint(filepath)
        else:
            raise ValueError("Model has not been trained yet. Please call fit() first.")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint
        """
        self.lightning_model = DeepChemLightningModule.load_from_checkpoint(
            filepath,
            model=self.model
        )
        return self.lightning_model
from deepchem.trans.transformers import DAGTransformer

# Example usage
if __name__ == "__main__":
    # Set random seed
    L.seed_everything(42)
    
    # Load dataset
    def test_dag_model():
        tasks, datasets, transformers = dc.molnet.load_bace_classification(
                "GraphConv", reload=False)
        train_dataset, _, test_dataset = datasets

        
        max_atoms = max([mol.get_num_atoms() for mol in train_dataset.X]+[mol.get_num_atoms() for mol in test_dataset.X])
        transformer = DAGTransformer(max_atoms=max_atoms)
        dataset = transformer.transform(train_dataset)
        model = dc.models.torch_models.DAGModel(len(tasks),
                        max_atoms=max_atoms,
                        mode='classification',
                        learning_rate=0.001,
                        batch_size=32)

        # Create trainer
        trainer = DeepChemLightningTrainer(
            model=model,
            batch_size=32,
            max_epochs=10,
            accelerator="cuda",
            devices=-1,
            log_every_n_steps=1,
            strategy="fsdp",
            fast_dev_run=True
        )
        
        # Train model
        trainer.fit(train_dataset)
        


        
        # Save checkpoint
        trainer.save_checkpoint("deepchem_model_dag.ckpt")

        
        transformer = DAGTransformer(max_atoms=max_atoms)
        dataset = transformer.transform(test_dataset)
        model = dc.models.torch_models.DAGModel(len(tasks),
                        max_atoms=max_atoms,
                        mode='classification',
                        learning_rate=0.001,
                        batch_size=32)
        
        trainer = DeepChemLightningTrainer(
            model=model,
            batch_size=32,
            max_epochs=10,
            accelerator="cuda",
            devices=-1,
            log_every_n_steps=1,
            strategy="fsdp",
        )

        trainer.load_checkpoint("deepchem_model_dag.ckpt")
        predictions = trainer.predict(test_dataset)
        print(predictions[0])

    def test_multitask_classifier():


        tasks, datasets, _ = dc.molnet.load_clintox()
        _, valid_dataset, _ = datasets

        model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                    n_features=1024,
                                    layer_sizes=[1000],
                                    dropouts=0.2,
                                    learning_rate=0.0001,
                                    device="cpu")
        trainer = DeepChemLightningTrainer(
                model=model,
                batch_size=32,
                max_epochs=30,
                accelerator="cuda",
                devices=-1,
                log_every_n_steps=1,
                strategy="fsdp",
                # fast_dev_run=True
            )
        trainer.fit(valid_dataset)
        trainer.save_checkpoint("multitask_classifier.ckpt")

        model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                    n_features=1024,
                                    layer_sizes=[1000],
                                    dropouts=0.2,
                                    learning_rate=0.0001,
                                    device="cpu")
        trainer = DeepChemLightningTrainer(
                model=model,
                batch_size=32,
                max_epochs=10,
                accelerator="cuda",
                devices=-1,
                log_every_n_steps=1,
                strategy="fsdp",
            )
        trainer.load_checkpoint("multitask_classifier.ckpt")
        predictions = trainer.predict(valid_dataset)
        print(predictions[0])


    test_multitask_classifier()