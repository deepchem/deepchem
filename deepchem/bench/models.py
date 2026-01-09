"""
Model registry for benchmark models.
"""
from typing import Dict, Tuple, Type, Any

from deepchem.models import GraphConvModel, MPNNModel, GAT, Model
from deepchem.models.torch_models import AttentiveFPModel
from deepchem.models.optimizers import Adam

class ModelRegistry:
    """Registry of available models for benchmarking."""

    def __init__(self):
        self._models = {
            "graphconv": (GraphConvModel, {
                "n_tasks": 1,
                "mode": "classification",
                "batch_size": 128,
                "learning_rate": 0.001
            }),
            "mpnn": (MPNNModel, {
                "n_tasks": 1,
                "mode": "classification", 
                "batch_size": 128,
                "learning_rate": 0.001
            }),
            "gat": (GAT, {
                "n_tasks": 1,
                "mode": "classification",
                "batch_size": 128,
                "learning_rate": 0.001
            }),
            "attentivefp": (AttentiveFPModel, {
                "n_tasks": 1,
                "mode": "classification",
                "batch_size": 128,
                "learning_rate": 0.001
            })
        }

    def get_model(self, name: str) -> Tuple[Type[Model], Dict[str, Any]]:
        """Get model class and default parameters.
        
        Parameters
        ----------
        name: str
            Name of model
            
        Returns
        -------
        Tuple[Type[Model], Dict[str, Any]]
            Model class and default parameters
        """
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        return self._models[name]

    def register_model(self, name: str, model_cls: Type[Model],
                      default_params: Dict[str, Any]):
        """Register a new model.
        
        Parameters
        ----------
        name: str
            Name to register model under
        model_cls: Type[Model] 
            Model class
        default_params: Dict[str, Any]
            Default parameters for model
        """
        self._models[name] = (model_cls, default_params)

    def list_models(self) -> Dict[str, Tuple[Type[Model], Dict[str, Any]]]:
        """Get dictionary of registered models."""
        return self._models.copy()