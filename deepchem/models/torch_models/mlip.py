import torch
import torch.nn as nn
import numpy as np
try:
    import dgl
except ImportError:
    pass
from deepchem.models.torch_models.torch_model import TorchModel


class _MLIPWrapper(nn.Module):
    """
    Internal wrapper to adapt TorchModel's list input to backbone's signature.
    TorchModel passes inputs as a single list argument. This wrapper unpacks it.
    """

    def __init__(self, module: nn.Module):
        super(_MLIPWrapper, self).__init__()
        self.module = module

    def forward(self, inputs):
        # Unpack the list [graph, atomic_numbers]
        g, z = inputs
        return self.module(g, z)


class MLIPModel(TorchModel):
    """
    A generic Machine Learning Interatomic Potential Wrapper.
    
    This model wraps an arbitrary backbone (e.g. NequIP, Allegro) that implements
    the MLIP signature (takes graph+atoms, outputs energy+forces) and provides
    standard DeepChem functionality:
    - Training Loop
    - Batching (DGL)
    - Composite Loss (Energy + Forces)
    
    Parameters
    ----------
    module : nn.Module
        The backbone neural network (e.g. NequIP).
    learning_rate : float
        Learning rate.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    """

    def __init__(self,
                 module: nn.Module,
                 learning_rate: float = 0.001,
                 energy_weight: float = 1.0,
                 force_weight: float = 100.0,
                 **kwargs):

        # Wrap the backbone to handle input unpacking
        wrapper = _MLIPWrapper(module)

        super(MLIPModel, self).__init__(wrapper,
                                        loss=self._mlip_loss,
                                        learning_rate=learning_rate,
                                        **kwargs)
        self.energy_weight = energy_weight
        self.force_weight = force_weight

    def _to_numpy(self, val):
        """
        Recursively converts Tensors/Lists/Arrays to clean Numpy Array.
        Handles:
        - Tensor (with/without grad) -> detach().cpu().numpy()
        - List of Tensors -> Array of values
        - Object Array wrapping Tensors -> Array of values
        """
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        # 2. List/Tuple Case
        if isinstance(val, (list, tuple)):
            sanitized = [self._to_numpy(v) for v in val]
            try:
                return np.array(sanitized)
            except ValueError:
                return np.array(sanitized, dtype=object)

        # 3. Object Array Case
        if isinstance(val, np.ndarray):
            if val.dtype == object:
                flat = val.flatten()
                if len(flat) > 0:
                    sanitized = [self._to_numpy(v) for v in flat]
                    try:
                        return np.array(sanitized).reshape(val.shape)
                    except ValueError:
                        return np.array(sanitized,
                                        dtype=object).reshape(val.shape)
            return val

        # 4. Standard Case (scalar/already numpy)
        try:
            return np.array(val)
        except ValueError:
            # Last resort for weird mixed types
            return np.array(val, dtype=object)

    def _mlip_loss(self, outputs, labels, weights):
        """
        Calculates the weighted sum of energy and force losses.
        """
        # 1. Robust Output Unpacking
        # outputs can be: (E, F) or [(E, F)] or [[E, F]]
        if isinstance(outputs, list) and len(outputs) == 1:
            outputs = outputs[0]

        if len(outputs) >= 2:
            pred_e, pred_f = outputs[0], outputs[1]
        else:
            # Fallback for unexpected return shapes
            pred_e, pred_f = outputs[0], None

        # 2. Robust Label Unpacking
        # labels are prepared by _prepare_batch to be clean Tensors
        target_e = labels[0]
        # Check if forces exist and are not empty
        target_f = labels[1] if len(labels) > 1 else None

        # 3. Loss Computation
        loss_e = torch.mean((pred_e - target_e)**2)

        loss_f = 0.0
        # Only compute force loss if both prediction and target exist
        if target_f is not None and pred_f is not None:
            loss_f = torch.mean((pred_f - target_f)**2)

        return self.energy_weight * loss_e + self.force_weight * loss_f

    def _prepare_batch(self, batch):
        """
        Prepare batch for training.
        """
        inputs, labels, weights = batch

        # 1. Prepare DGL Graphs
        dgl_graphs = [g.to_dgl_graph() for g in inputs[0]]
        g_batch = dgl.batch(dgl_graphs).to(self.device)

        # Handle Node Features (Atomic Numbers)
        features = g_batch.ndata['x']
        atomic_numbers = features[:, -1].long()

        # Set pos requires_grad for force computation
        if 'pos' in g_batch.ndata:
            g_batch.ndata['pos'].requires_grad_(True)

        # 2. Universal Label Sanitizer
        if labels is not None and len(labels) > 0:

            import traceback

            # `labels` from generator is `[y_b]`. `y_b` is the batch of labels.
            y_data = labels[0]

            if y_data is not None:
                # Handle 3D labels logic (Batch, 1, Tasks) -> (Batch, Tasks)
                # Some DeepChem splitters produce this shape
                if hasattr(
                        y_data,
                        'ndim') and y_data.ndim == 3 and y_data.shape[1] == 1:
                    y_data = y_data[:, 0]

                new_labels = []

                # Temporary storage
                e_vals = []
                f_vals = []
                has_forces = False

                try:
                    for i, l in enumerate(y_data):
                        # l can be:
                        # - scalar (Energy)
                        # - [Energy]
                        # - [Energy, Forces]
                        # - np.array([Energy, Forces])

                        # Convert l to a list/sequence if it claims to be one, else treat as scalar E
                        e_item = None
                        f_item = None

                        is_sequence = False
                        if not isinstance(l, (str, bytes)):
                            # Check if it has length. 0-d arrays have NO length.
                            try:
                                _ = len(l)
                                is_sequence = True
                            except (TypeError, IndexError):
                                # It is a scalar or 0-d array
                                is_sequence = False

                        if is_sequence:
                            if len(l) == 0:
                                # Empty label?
                                e_item = 0.0
                            elif len(l) >= 1:
                                e_item = l[0]

                            if len(l) >= 2:
                                f_item = l[1]
                                has_forces = True
                        else:
                            # Scalar
                            e_item = l

                        # --- Process Energy ---
                        e_clean = self._to_numpy(e_item)
                        # Flatten if it's an array
                        if hasattr(e_clean, 'shape'):
                            e_clean = e_clean.reshape(-1)
                            if e_clean.size > 0:
                                e_clean = float(e_clean[0])
                            else:
                                e_clean = 0.0  # Fallback
                        else:
                            e_clean = float(e_clean)
                        e_vals.append(e_clean)

                        # --- Process Forces ---
                        if has_forces and f_item is not None:
                            f_clean = self._to_numpy(f_item)
                            # Safe Cast
                            f_vals.append(f_clean)

                    # Create Energy Tensor
                    energy_batch = torch.tensor(e_vals,
                                                dtype=torch.float32,
                                                device=self.device)
                    new_labels.append(energy_batch)

                    # Create Force Tensor
                    if has_forces and len(f_vals) == len(e_vals):
                        # We need to stack them.
                        # f_vals is a list of arrays (N_atoms, 3).
                        # Concatenate them along dim 0.
                        # Verify shapes match graph sizes if possible, but for now just concat.
                        try:
                            forces_concat = np.concatenate(f_vals, axis=0)
                            forces_batch = torch.tensor(forces_concat,
                                                        dtype=torch.float32,
                                                        device=self.device)
                            new_labels.append(forces_batch)
                        except ValueError as e:
                            # If shapes don't match or f_vals is empty
                            print(f"Warning: Force concatenation failed: {e}")

                    # Replace labels with the clean tensors
                    labels = new_labels

                except Exception as e:
                    # If sanitizer fails, we MUST raise error or return valid tensors (even if empty)
                    print(f"Error preparing MLIP labels: {e}")
                    traceback.print_exc()
                    # Stop iteration and raise to alert user
                    raise e

        # Standard Tuple for TorchModel: (Inputs, Labels, Weights)
        return [g_batch, atomic_numbers], labels, weights

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=False):
        """
        Creates a generator that iterates batches for a dataset.
        
        Overrides TorchModel.default_generator to force pad_batches=False.
        MLIP models deal with ragged inputs (different graph sizes, different force vectors)
        that NumpyDataset.pad_batch() essentially corrupts by trying to stack them.
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=False):
                yield ([X_b], [y_b], [w_b])
