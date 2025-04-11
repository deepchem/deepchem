"""
Unit tests for the PINO model.

These tests cover model initialization, multi-resolution input handling,
boundary condition enforcement, physics compliance, parameter learning, and autograd behavior.

References:
    "Fourier Neural Operator for Parametric Partial Differential Equations"
    (https://arxiv.org/abs/2111.03794)
"""

import pytest
import unittest
import numpy as np
import torch
import deepchem as dc  # noqa: F401 -- Not used directly here
from deepchem.data import NumpyDataset
from deepchem.models.torch_models.pino import PINO

# The following try/except is redundant since torch is already imported.
# Remove it if not needed, or suppress the redefinition warning:
try:
    import torch  # noqa: F811 -- redundant import, but kept for compatibility if needed

    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
class TestPINO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = np.linspace(-1, 1, 100)
        t = np.linspace(0, 1, 1)
        X, T = np.meshgrid(x, t)
        cls.inputs = np.stack([X.ravel(), T.ravel()], axis=1)  # shape (100, 2)
        cls.outputs = np.sin(np.pi * X).ravel()[:, np.newaxis]  # shape (100, 1)
        boundary_points = np.concatenate(
            [
                np.column_stack((np.full_like(t, -1.0), t)),
                np.column_stack((np.full_like(t, 1.0), t)),
            ]
        )
        cls.boundary_data = {
            "dirichlet": {
                "points": torch.as_tensor(
                    boundary_points, dtype=torch.float32, device=cls.device
                ),
                "values": torch.zeros(
                    (boundary_points.shape[0], 1),
                    dtype=torch.float32,
                    device=cls.device,
                ),
            }
        }

        def burgers_residual(u_t, u_x, u_xx, outputs, inputs, nu):
            return u_t + outputs * u_x - nu * u_xx

        cls.burgers_residual = burgers_residual

        cls.common_args = {
            "in_channels": 2,
            "modes": 16,
            "width": 64,
            "pde_fn": cls.burgers_residual,
            "boundary_data": cls.boundary_data,
            "boundary_weight": 5.0,
            "data_weight": 1.0,
            "physics_weight": 0.1,
            "hi_res": (64,),
            "learning_rate": 0.0001,
            "phys_params": {
                "alpha": torch.nn.Parameter(
                    torch.tensor(np.log(0.1), dtype=torch.float32, device=cls.device)
                )
            },
        }
        cls.dataset = NumpyDataset(cls.inputs, cls.outputs)
        print("[TestPINO] setUpClass complete.")

    @pytest.mark.torch
    def test_pino_initialization(self):
        print("[TestPINO] test_pino_initialization started.")
        model = PINO(**self.common_args)
        self.assertIsInstance(model.model, PINO.FNO1D)
        self.assertEqual(len(model.model.fourier_blocks), 4)
        print("[TestPINO] test_pino_initialization passed.")

    @pytest.mark.torch
    def test_multi_resolution_handling(self):
        print("[TestPINO] test_multi_resolution_handling started.")
        batch = [
            {
                "input": torch.as_tensor(self.inputs, dtype=torch.float32),
                "label": torch.as_tensor(self.outputs, dtype=torch.float32),
            }
        ]
        inputs, labels, weights = PINO(**self.common_args)._prepare_batch(batch)
        self.assertTrue(isinstance(inputs, torch.Tensor))
        self.assertEqual(inputs.ndim, 3)
        if labels is not None:
            self.assertTrue(isinstance(labels, torch.Tensor))
        print("[TestPINO] test_multi_resolution_handling passed.")

    @pytest.mark.torch
    def test_boundary_condition_enforcement(self):
        print("[TestPINO] test_boundary_condition_enforcement started.")
        model = PINO(**self.common_args)
        loss_history = model.fit(self.dataset, nb_epoch=5, phys_lr=1e-2)  # noqa: F841
        boundary_pts = self.boundary_data["dirichlet"]["points"]
        pred_boundaries = model.model(boundary_pts)
        boundary_loss = torch.mean(
            (pred_boundaries - self.boundary_data["dirichlet"]["values"]) ** 2
        )
        self.assertLess(boundary_loss.item(), 1.0)
        print("[TestPINO] test_boundary_condition_enforcement passed.")

    @pytest.mark.torch
    def test_physics_compliance(self):
        print("[TestPINO] test_physics_compliance started.")
        model = PINO(**self.common_args)
        loss_history = model.fit(self.dataset, nb_epoch=3, phys_lr=1e-2)  # noqa: F841
        self.assertGreater(len(loss_history), 0)
        initial_loss = loss_history[0]
        loss_history = model.fit(self.dataset, nb_epoch=50, phys_lr=1e-2)
        final_loss = loss_history[-1]
        print(f"[TestPINO] initial_loss: {initial_loss}, final_loss: {final_loss}")
        self.assertLess(final_loss, initial_loss)
        print("[TestPINO] test_physics_compliance passed.")

    @pytest.mark.torch
    def test_parameter_learning(self):
        print("[TestPINO] test_parameter_learning started.")
        args = self.common_args.copy()
        # Increase physics_weight significantly to boost the influence of the PDE loss on alpha.
        args["physics_weight"] = 10
        # Initialize alpha with log(0.05)
        args["phys_params"] = {
            "alpha": torch.nn.Parameter(
                torch.tensor(np.log(0.05), dtype=torch.float32, device=self.device)
            )
        }
        model = PINO(**args)
        loss_history = model.fit(self.dataset, nb_epoch=200, phys_lr=1e-2)  # noqa: F841
        # Learned nu is computed as exp(alpha)
        learned_nu = torch.exp(model.phys_params["alpha"]).item()
        print("Learned nu:", learned_nu)
        self.assertTrue(0.005 < learned_nu < 0.05)
        print("[TestPINO] test_parameter_learning passed.")

    @pytest.mark.torch
    def test_autograd_physics_residual(self):
        print("[TestPINO] test_autograd_physics_residual started.")
        model = PINO(**self.common_args)
        test_input = torch.tensor(
            [[0.5, 0.1]], dtype=torch.float32, device=self.device, requires_grad=True
        )
        output = model.model(test_input)
        loss = model._compute_pde_loss(test_input, output)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(test_input.grad)
        print("[TestPINO] test_autograd_physics_residual passed.")


# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
