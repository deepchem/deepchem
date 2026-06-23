import torch
import torch.nn as nn
import deepchem as dc
from deepchem.models.torch_models import TorchModel
from typing import List, Optional, Union, Dict

class OntologyNode:
    """
    A node in the ontology graph.
    """
    def __init__(self, id, n_outputs, feature_ids=None, children=None):
        self.id = id
        self.n_outputs = n_outputs
        self.feature_ids = feature_ids if feature_ids is not None else []
        self.children = children if children is not None else []

class OntologyNN(nn.Module):
    """
    The internal PyTorch Module for OntologyModel.
    """
    def __init__(self, n_tasks, feature_ids, root_node, n_classes=2):
        super(OntologyNN, self).__init__()
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.feature_ids = feature_ids
        self.root_node = root_node
        
        # Map feature names to indices for quick lookup
        self.feature_index = {f: i for i, f in enumerate(feature_ids)}
        
        # Registry to hold layers for each node
        self.node_layers = nn.ModuleDict()
        self.prediction_heads = nn.ModuleDict()
        
        # Recursively build the layers
        self._build_layers(root_node)

    def _build_layers(self, node):
        """Recursively construct layers for the graph."""
        # 1. Calculate Input Dimension
        input_dim = len(node.feature_ids)
        
        for child in node.children:
            # Ensure child layers exist
            if str(child.id) not in self.node_layers:
                self._build_layers(child)
            # Add child's output size to this node's input size
            input_dim += child.n_outputs
            
        # 2. Define the main block for this node (Dense -> BatchNorm -> Tanh)
        self.node_layers[str(node.id)] = nn.Sequential(
            nn.Linear(input_dim, node.n_outputs),
            nn.BatchNorm1d(node.n_outputs),
            nn.Tanh()
        )
        
        # 3. Define the prediction head for this node (for intermediate loss)
        self.prediction_heads[str(node.id)] = nn.Linear(node.n_outputs, self.n_tasks)

    def forward(self, x):
        # We need to store outputs of every node to handle the graph connections
        node_outputs = {}
        all_predictions = {}
        
        # We need a helper to traverse recursively at runtime
        def process_node(node):
            if str(node.id) in node_outputs:
                return node_outputs[str(node.id)]
            
            # 1. Gather inputs from children
            child_outputs = []
            for child in node.children:
                child_outputs.append(process_node(child))
            
            # 2. Gather inputs from features (x)
            feature_indices = [self.feature_index[fid] for fid in node.feature_ids]
            current_inputs = []
            
            if feature_indices:
                f_idx = torch.tensor(feature_indices, device=x.device)
                features = torch.index_select(x, 1, f_idx)
                current_inputs.append(features)
                
            # Combine features + children
            current_inputs.extend(child_outputs)
            combined_input = torch.cat(current_inputs, dim=1)
            
            # 3. Pass through this node's layer
            out = self.node_layers[str(node.id)](combined_input)
            node_outputs[str(node.id)] = out
            
            # 4. Compute prediction for this node (for loss)
            pred = self.prediction_heads[str(node.id)](out)
            all_predictions[str(node.id)] = pred
            
            return out

        # Run the graph
        process_node(self.root_node)
        
        # --- RETURN LOGIC ---
        if self.training:
            # Training Mode: Return DICT (for multi-node loss)
            return all_predictions
        else:
            # Evaluation/Prediction Mode: Return TENSOR (root only)
            return all_predictions[str(self.root_node.id)]

class _OntologyLoss(nn.Module):
    """
    Custom loss function for OntologyModel.
    """
    def __init__(self, root_id, intermediate_weight, mode):
        super(_OntologyLoss, self).__init__()
        self.root_id = str(root_id)
        self.intermediate_weight = intermediate_weight
        self.mode = mode
        if mode == 'regression':
            self.base_loss = nn.MSELoss(reduction='none') 
        else:
            self.base_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, labels, weights):
        total_loss = 0.0
        
        # --- FIX: Unwrap DeepChem list wrappers ---
        if isinstance(labels, list):
            labels = labels[0]
            
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
            
        # Ensure labels are on the correct device
        device = next(iter(outputs.values())).device
        labels = labels.to(device)
        
        if self.mode == 'regression':
            labels = labels.float()
        else:
            labels = labels.long()

        # Handle weights unwrap + conversion
        if weights is not None:
            if isinstance(weights, list):
                weights = weights[0]
            if not isinstance(weights, torch.Tensor):
                weights = torch.as_tensor(weights)
            weights = weights.to(device)

        for node_id, prediction in outputs.items():
            loss_scale = 1.0 if node_id == self.root_id else self.intermediate_weight
            
            node_loss = self.base_loss(prediction, labels)
            
            if weights is not None:
                if weights.shape != node_loss.shape:
                    weights = weights.view_as(node_loss)
                node_loss = node_loss * weights
                
            total_loss = total_loss + (node_loss.mean() * loss_scale)
            
        return total_loss

class OntologyModel(TorchModel):
    """
    PyTorch implementation of OntologyModel.
    """
    def __init__(self, n_tasks, feature_ids, root_node, mode="regression", 
                 n_classes=2, intermediate_loss_weight=0.3, **kwargs):
        
        # 1. Initialize the internal PyTorch Module
        module = OntologyNN(n_tasks, feature_ids, root_node, n_classes)
        
        # 2. Define the custom loss function
        criterion = _OntologyLoss(root_node.id, intermediate_loss_weight, mode)
        
        # 3. Initialize the parent TorchModel
        super(OntologyModel, self).__init__(
            module, loss=criterion, **kwargs
        )
        self.root_id = str(root_node.id)