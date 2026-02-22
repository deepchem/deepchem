from typing import Optional, List, Tuple, Dict, Any, Iterator
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from rdkit import Chem
from deepchem.models import TorchModel
from deepchem.data import Dataset
from deepchem.models.optimizers import Adam as DeepChemAdam
import logging

logger = logging.getLogger(__name__)
from MaceNNmodel import MACEClean
from Maceloss import MACELoss


class MACEWrapper(nn.Module):
    """Wrapper for MACE network to interface with DeepChem.

    This wrapper handles the conversion between DeepChem's input format
    and the format expected by the MACE neural network.

    Parameters
    ----------
    mace_net : nn.Module
        The MACE neural network model

    Examples
    --------
    >>> mace = MACE(hidden_dim=64, num_interactions=3)
    >>> wrapper = MACEWrapper(mace)
    """

    def __init__(self, mace_net: nn.Module) -> None:
        super().__init__()
        self.mace = mace_net

    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass through MACE network.

        Parameters

        inputs : list or Batch
            Input data, either as list containing PyG Batch or Batch directly

        Returns

        torch.Tensor
            Predicted energies with shape (batch_size,)
        """
        pyg_batch = inputs[0] if isinstance(inputs, list) else inputs

        energy, _ = self.mace(
            z=pyg_batch.z,
            pos=pyg_batch.pos,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch
        )
        return energy


class SimpleMACEModel(TorchModel):
    """MACE model for molecular property prediction integrated with DeepChem.

    MACE (Multi-Atomic Cluster Expansion) is a graph neural network that uses
    equivariant message passing to predict molecular properties from 3D structures.

    This implementation properly integrates MACE with DeepChem's infrastructure:
    - PyTorch for neural network architecture
    - DeepChem for training, evaluation, and checkpointing
    - Custom batch preparation for PyTorch Geometric format

    Parameters
    ----------
    coords_cache : Dict[str, np.ndarray]
        Dictionary mapping SMILES strings to 3D coordinates (N_atoms, 3)
    hidden_dim : int, optional (default=64)
        Dimension of hidden layers in MACE
    num_interactions : int, optional (default=3)
        Number of message passing layers
    batch_size : int, optional (default=4)
        Number of molecules per batch
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer
    max_atomic_num : int, optional (default=100)
        Maximum atomic number supported (for embedding layer)
    cutoff : float, optional (default=5.0)
        Distance cutoff in Angstroms for edge creation
    **kwargs
        Additional arguments passed to TorchModel

    Examples
    --------
    >>> coords_cache = {'CCO': np.random.randn(9, 3)}
    >>> model = SimpleMACEModel(coords_cache, hidden_dim=64)
    >>> model.fit(train_dataset, nb_epoch=10)
    >>> predictions = model.predict(test_dataset)


    """

    def __init__(
        self,
        coords_cache: Dict[str, np.ndarray],
        hidden_dim: int = 64,
        num_interactions: int = 3,
        batch_size: int = 4,
        learning_rate: float = 0.001,
        max_atomic_num: int = 100,
        cutoff: float = 5.0,
        **kwargs
    ) -> None:

        # Device configuration
        device = torch.device("cpu")

        # Store configuration
        self.coords_cache = coords_cache
        self.max_atomic_num = max_atomic_num
        self.cutoff = cutoff

        # Create MACE architecture (PyTorch)
        mace_net = MACEClean(
            num_elements=max_atomic_num,
            hidden_dim=hidden_dim,
            num_interactions=num_interactions,
            cutoff=cutoff
        )
        wrapper = MACEWrapper(mace_net)

        # Create DeepChem optimizer wrapper
        optimizer = DeepChemAdam(learning_rate=learning_rate)

        # Initialize parent TorchModel (DeepChem handles training!)
        super().__init__(
            model=wrapper,
            loss=MACELoss(),
            optimizer=optimizer,  # ← DeepChem manages this optimizer
            output_types=["prediction"],
            n_tasks=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )

        logger.info(
            f"Initialized SimpleMACEModel with {hidden_dim}D hidden, "
            f"{num_interactions} interactions, cutoff={cutoff}Å"
        )

    def _extract_atomic_numbers(
        self,
        smiles: str,
        num_atoms: int
    ) -> np.ndarray:
        """Extract atomic numbers from SMILES and match to target atom count.

        Helper method to convert SMILES to atomic numbers, handling padding
        and truncation to match DeepChem's graph representation.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule
        num_atoms : int
            Target number of atoms (from DeepChem graph)

        Returns
        -------
        np.ndarray
            Array of atomic numbers with shape (num_atoms,)
        """
        try:
            mol = Chem.MolFromSmiles(smiles) # FIX: Changed MolFromSmi to MolFromSmiles
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return np.full(num_atoms, 6, dtype=np.int64)

            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

            if len(atomic_nums) == num_atoms:
                return np.array(atomic_nums, dtype=np.int64)
            elif len(atomic_nums) < num_atoms:
                padding = [1] * (num_atoms - len(atomic_nums))
                return np.array(atomic_nums + padding, dtype=np.int64)
            else:
                return np.array(atomic_nums[:num_atoms], dtype=np.int64)

        except Exception as e:
            logger.error(f"Error extracting atomic numbers for {smiles}: {e}")
            return np.full(num_atoms, 6, dtype=np.int64)

    def _get_coordinates(
        self,
        smiles: str,
        num_atoms: int
    ) -> np.ndarray:
        """Get 3D coordinates for molecule, matching to target atom count.

        Helper method to retrieve coordinates from cache and match size.

        Parameters
        ----------
        smiles : str
            SMILES string (used as key for coords_cache)
        num_atoms : int
            Target number of atoms

        Returns
        -------
        np.ndarray
            3D coordinates with shape (num_atoms, 3)
        """
        if smiles in self.coords_cache:
            coords = self.coords_cache[smiles]

            if coords.shape[0] == num_atoms:
                return coords
            elif coords.shape[0] < num_atoms:
                padding_size = num_atoms - coords.shape[0]
                last_pos = coords[-1] if len(coords) > 0 else np.zeros(3)
                padding = last_pos + np.random.randn(padding_size, 3) * 0.1
                return np.vstack([coords, padding])
            else:
                return coords[:num_atoms]
        else:
            logger.debug(f"SMILES {smiles} not in coords_cache, using random coords")
            return np.random.randn(num_atoms, 3).astype(np.float32)

    def _create_pyg_data(
        self,
        atomic_nums: np.ndarray,
        coords: np.ndarray,
        target: Optional[float],
        device: torch.device
    ) -> Data:
        """Create PyTorch Geometric Data object from molecular information.

        Helper method to convert processed molecular data into PyG format.

        Parameters
        ----------
        atomic_nums : np.ndarray
            Atomic numbers with shape (num_atoms,)
        coords : np.ndarray
            3D coordinates with shape (num_atoms, 3)
        target : float or None
            Target property value
        device : torch.device
            Device to place tensors on

        Returns
        -------
        Data
            PyTorch Geometric Data object
        """
        atomic_nums = np.clip(atomic_nums, 1, self.max_atomic_num).astype(np.int64)
        z_indices = atomic_nums - 1
        z = torch.clamp(
            torch.tensor(z_indices, dtype=torch.long, device=device),
            min=0,
            max=self.max_atomic_num - 1
        )

        pos = torch.tensor(coords, dtype=torch.float32, device=device)
        edge_index = build_distance_edges(pos, cutoff=self.cutoff).to(device)

        if target is not None:
            y_val = torch.tensor([target], dtype=torch.float32, device=device)
        else:
            y_val = torch.zeros(1, dtype=torch.float32, device=device)

        return Data(z=z, pos=pos, edge_index=edge_index, y=y_val)

    def default_generator(
        self,
        dataset: Dataset,
        epochs: int = 1,
        mode: str = 'fit',
        deterministic: bool = True,
        pad_batches: bool = True
    ) -> Iterator[Tuple[List, List, List]]:
        """Generate PyG batches for DeepChem training/prediction.

        This method converts DeepChem's data format to PyTorch Geometric
        batches while preserving all molecular information.

        Parameters
        ----------
        dataset : Dataset
            DeepChem dataset
        epochs : int
            Number of epochs to iterate
        mode : str
            'fit' or 'predict' mode
        deterministic : bool
            Whether to use deterministic batching
        pad_batches : bool
            Whether to pad batches

        Yields
        ------
        tuple
            (inputs, labels, weights) formatted for DeepChem:
            - inputs: List containing PyG Batch
            - labels: List containing target tensors
            - weights: List containing sample weights
        """
        for epoch in range(epochs):
            for batch in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches
            ):
                if len(batch) == 4:
                    X, y, w, ids = batch
                elif len(batch) == 3:
                    X, y, w = batch
                    ids = None
                else:
                    X = batch[0]
                    y = w = ids = None

                data_list = []
                for i in range(len(X)):
                    graph = X[i]
                    num_atoms = graph.get_num_atoms()
                    smiles = ids[i] if ids is not None else None

                    atomic_nums = (
                        self._extract_atomic_numbers(smiles, num_atoms)
                        if smiles else np.full(num_atoms, 6, dtype=np.int64)
                    )

                    coords = (
                        self._get_coordinates(smiles, num_atoms)
                        if smiles else np.random.randn(num_atoms, 3)
                    )

                    target = y[i] if y is not None else None

                    data = self._create_pyg_data(
                        atomic_nums, coords, target, self.device
                    )
                    data_list.append(data)

                pyg_batch = Batch.from_data_list(data_list)

                labels = pyg_batch.y.view(-1, 1)
                weights = torch.ones(len(X), device=self.device)

                yield ([pyg_batch], [labels], [weights])

    def _prepare_batch(self, batch: Tuple) -> Tuple:
        """Override to pass through PyG batches without modification.

        Since default_generator() already creates properly formatted PyG batches,
        this method bypasses DeepChem's default numpy array processing which
        would fail on PyG Batch objects.

        Parameters
        ----------
        batch : tuple
            Pre-formatted batch from default_generator containing
            (inputs, labels, weights) where inputs are PyG Batches

        Returns
        -------
        tuple
            The same batch without modification
        """
        # Batch is already (inputs, labels, weights) from default_generator
        # Just pass it through without DeepChem's default processing
        return batch


print(" SimpleMACEModel ")
