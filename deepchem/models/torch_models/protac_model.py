import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import Adam as DChem_Adam
from layers import GraphConv, SmilesNet


class ProtacModel(nn.Module):
    def __init__(self, 
                 ligase_ligand_model, 
                 ligase_pocket_model,
                 target_ligand_model, 
                 target_pocket_model, 
                 smiles_model):
        
        super().__init__()
        self.ligase_ligand_model = ligase_ligand_model
        self.ligase_pocket_model = ligase_pocket_model
        self.target_ligand_model = target_ligand_model
        self.target_pocket_model = target_pocket_model
        self.smiles_model = smiles_model
        self.fc1 = nn.Linear(64*5,64)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(64,2)

    def forward(self, inputs):
        # Handle inputs as a tuple/list or unpack arguments
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if len(inputs) == 6:
                ligase_ligand, ligase_pocket, target_ligand, target_pocket, smiles, smiles_length = inputs
            else:
                raise ValueError(f"Expected 6 inputs, got {len(inputs)}")
        else:
            # Single input case - this shouldn't happen with your setup
            raise ValueError("Expected tuple or list of inputs, got single input")
        
        v_0 = self.ligase_ligand_model(ligase_ligand)
        v_1 = self.ligase_pocket_model(ligase_pocket)
        v_2 = self.target_ligand_model(target_ligand)
        v_3 = self.target_pocket_model(target_pocket)
        v_4 = self.smiles_model(smiles, smiles_length)
        v_f = torch.cat((v_0, v_1, v_2, v_3, v_4), 1)
        v_f = self.relu(self.fc1(v_f))
        v_f = self.fc2(v_f)
        return v_f


# Custom loss function that can handle weights
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, outputs, labels, weights=None):
        # Calculate regular cross entropy loss
        loss = self.cross_entropy(outputs, labels)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Return mean loss
        return loss.mean()


class DeepPROTAC(TorchModel):
    def __init__(self, **kwargs):
        # Initialize the individual models
        ligase_ligand = GraphConv(num_embeddings=10)  
        ligase_pocket = GraphConv(num_embeddings=5)   
        target_ligand = GraphConv(num_embeddings=10)  
        target_pocket = GraphConv(num_embeddings=5)   
        smiles = SmilesNet(batch_size=1)              
        model = ProtacModel(
            ligase_ligand_model=ligase_ligand,
            ligase_pocket_model=ligase_pocket,
            target_ligand_model=target_ligand,
            target_pocket_model=target_pocket,
            smiles_model=smiles,
        )
        # Use our custom weighted loss function
        loss = WeightedCrossEntropyLoss()
        # Initialize the TorchModel with DeepChem's optimizer
        optimizer = DChem_Adam(learning_rate=0.001)
        super().__init__(model=model, loss=loss, optimizer=optimizer, output_types=['prediction'], **kwargs)
        # Create the PyTorch optimizer directly
        self._pytorch_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _prepare_batch(self, batch):
        """Override this method to handle custom batch format."""
        # Check if batch is already in the right format (inputs, labels, weights)
        if isinstance(batch, tuple) and len(batch) == 3:
            inputs, labels, weights = batch
        else:
            # If it's just the inputs (like during prediction)
            inputs = batch
            labels = None
            weights = None
            
        # If inputs is a dictionary, convert to tuple for the model
        if isinstance(inputs, dict):
            inputs_tuple = (
                inputs['ligase_ligand'],
                inputs['ligase_pocket'],
                inputs['target_ligand'],
                inputs['target_pocket'],
                inputs['smiles'],
                inputs['smiles_length'],
            )
        else:
            inputs_tuple = inputs

        # Convert labels to torch.long for CrossEntropyLoss if they exist
        if labels is not None:
            labels = labels.long()

        return inputs_tuple, labels, weights

    def default_generator(self, dataset, epochs=1, mode='fit', deterministic=True, pad_batches=True):
        """Override this method to handle PyTorch DataLoader."""
        if isinstance(dataset, torch.utils.data.DataLoader):
            # If the dataset is a PyTorch DataLoader, yield batches directly
            for epoch in range(epochs):
                for batch in dataset:
                    yield batch
        else:
            # If it's not a DataLoader, use the default behavior
            yield from super().default_generator(dataset, epochs, mode, deterministic, pad_batches)
            
    def fit_generator(self, generator, max_checkpoints_to_keep=5, checkpoint_interval=1000, restore=False, variables=None, loss=None, callbacks=[], all_losses=[]):
        """Override fit_generator to handle our custom training loop properly."""
        # Create a PyTorch optimizer if it doesn't exist
        if not hasattr(self, '_pytorch_optimizer'):
            self._pytorch_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Track the number of batches processed
        batch_count = 0
        
        # Get the loss function (use passed loss or model's loss)
        loss_fn = loss if loss is not None else self.loss
        
        # Training loop
        for batch in generator:
            # Prepare the batch data
            inputs, labels, weights = self._prepare_batch(batch)
            
            # Zero the gradients
            self._pytorch_optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss - using our custom loss function that handles weights
            if weights is not None:
                batch_loss = loss_fn(outputs, labels, weights)
            else:
                batch_loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            batch_loss.backward()
            self._pytorch_optimizer.step()
            
            # Increment batch counter
            batch_count += 1
            
            # Save checkpoint if needed
            if checkpoint_interval > 0 and batch_count % checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Return the model itself
        return self
        
    def predict(self, dataset, transformers=None, output_types=None):
        """
        Override predict method to directly handle our custom dataset format
        
        Parameters
        ----------
        dataset: torch.utils.data.DataLoader
            DataLoader yielding batches of data
        transformers: List
            List of transformers to apply to outputs
        output_types: List
            List of output types to return
            
        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        self.model.eval()
        device = self.device
        
        all_outputs = []
        
        with torch.no_grad():
            for batch in dataset:
                # Handle both tuple format (inputs, labels, weights) and dictionary format
                if isinstance(batch, tuple) and len(batch) == 3:
                    inputs, _, _ = batch  # Unpack if in tuple format
                else:
                    inputs = batch  # Otherwise use as is
                
                # Prepare the inputs - convert dict to tuple if needed
                if isinstance(inputs, dict):
                    inputs_tuple = (
                        inputs['ligase_ligand'].to(device),
                        inputs['ligase_pocket'].to(device),
                        inputs['target_ligand'].to(device),
                        inputs['target_pocket'].to(device),
                        inputs['smiles'].to(device),
                        inputs['smiles_length']
                    )
                else:
                    inputs_tuple = inputs
                
                # Forward pass
                outputs = self.model(inputs_tuple)
                all_outputs.append(outputs.cpu())
        
        # Concatenate all outputs
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # Apply transformers if needed
        if transformers:
            outputs = self.transform_outputs(all_outputs, transformers)
        else:
            outputs = all_outputs.numpy()
            
        return outputs