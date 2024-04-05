"""Advantage Actor-Critic (A2C) algorithm for reinforcement learning."""
import numpy as np
from typing import List
try:
    import torch
    has_pytorch = True
except:
    has_pytorch = False


class A2CLossDiscrete(object):
    """This class computes the loss function for A2C with discrete action spaces.
    The A2C algorithm optimizes all outputs at once using a loss that is the sum of three terms:

    1. The policy loss, which seeks to maximize the discounted reward for each action.

    2. The value loss, which tries to make the value estimate match the actual discounted reward that was attained at each step.

    3. An entropy term to encourage exploration.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn.functional as F
    >>> outputs = [torch.tensor([[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]), torch.tensor([0.], requires_grad = True)]
    >>> labels = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype = np.float32)
    >>> discount = np.array([-1.0203744, -0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097, -0.9901, 0.01, -1. , 0. ], dtype=np.float32)
    >>> advantage = np.array([-1.0203744 ,-0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097, -0.9901 ,0.01 ,-1. , 0.], dtype = np.float32)
    >>> loss = A2CLossDiscrete(value_weight = 1.0, entropy_weight = 0.01, action_prob_index = 0, value_index = 1)
    >>> loss_val = loss(outputs, [labels], [discount, advantage])
    >>> loss_val
    tensor(1.2541, grad_fn=<SubBackward0>)
    """

    def __init__(self, value_weight: float, entropy_weight: float,
                 action_prob_index: int, value_index: int):
        """Computes the loss function for the A2C algorithm with discrete action spaces.

        Parameters
        ----------
        value_weight: float
            a scale factor for the value loss term in the loss function
        entropy_weight: float
            a scale factor for the entropy term in the loss function
        action_prob_index: int
            Index of the action probabilities in the model's outputs.
        value_index: int
            Index of the value estimate in the model's outputs.
        """
        self.value_weight: float = value_weight
        self.entropy_weight: float = entropy_weight
        self.action_prob_index: int = action_prob_index
        self.value_index: int = value_index

    def __call__(self, outputs: List[torch.Tensor], labels: List[np.ndarray],
                 weights: List[np.ndarray]):
        prob_array: torch.Tensor = outputs[self.action_prob_index]
        value: torch.Tensor = outputs[self.value_index]
        reward, advantage_array = weights
        action: torch.Tensor = torch.from_numpy(labels[0])
        advantage: torch.Tensor = torch.unsqueeze(
            torch.from_numpy(advantage_array), 1)
        prob: torch.Tensor = prob_array + torch.finfo(torch.float32).eps
        log_prob: torch.Tensor = torch.log(prob)
        policy_loss: torch.Tensor = -torch.mean(
            advantage * torch.sum(action * log_prob, 1))
        value_loss: torch.Tensor = torch.mean(
            torch.square(torch.from_numpy(reward) - value))
        entropy: torch.Tensor = -torch.mean(torch.sum(prob * log_prob, 1))
        return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


# Note: For continuous case, when an additional loss term correspoding to mean is calculated only then the gradients get calculated for the layers of the model.
class A2CLossContinuous(object):
    """This class computes the loss function for A2C with continuous action spaces.

    Example
    -------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn.functional as F
    >>> outputs = [torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], dtype=torch.float32, requires_grad=True), torch.tensor([10.], dtype=torch.float32, requires_grad=True), torch.tensor([[27.717865],[28.860144]], dtype=torch.float32, requires_grad=True)]
    >>> labels = np.array([[-4.897339 ], [ 3.4308329], [-4.527725 ], [-7.3000813], [-1.9869075], [20.664988 ], [-8.448957 ], [10.580486 ], [10.017258 ], [17.884674 ]], dtype=np.float32)
    >>> discount = np.array([4.897339, -8.328172, 7.958559, 2.772356, -5.313174, -22.651896, 29.113945, -19.029444, 0.56322646, -7.867417], dtype=np.float32)
    >>> advantage = np.array([-5.681633, -20.57494, -1.4520378, -9.348538, -18.378199, -33.907513, 25.572464, -32.485718 , -6.412546, -15.034998], dtype=np.float32)
    >>> loss = A2CLossContinuous(value_weight = 1.0, entropy_weight = 0.01, mean_index = 0, std_index = 1, value_index = 2)
    >>> loss_val = loss(outputs, [labels], [discount, advantage])
    >>> loss_val
    tensor(1050.2310, grad_fn=<SubBackward0>)
    """

    def __init__(self, value_weight: float, entropy_weight: float,
                 mean_index: int, std_index: int, value_index: int):
        """Computes the loss function for the A2C algorithm with continuous action spaces.

        Parameters
        ----------
        value_weight: float
            a scale factor for the value loss term in the loss function
        entropy_weight: float
            a scale factor for the entropy term in the loss function
        mean_index: int
            Index of the mean of the action distribution in the model's outputs.
        std_index : int
            Index of the standard deviation of the action distribution in the model's outputs.
        value_index: int
            Index of the value estimate in the model's outputs.
        """
        self.value_weight: float = value_weight
        self.entropy_weight: float = entropy_weight
        self.mean_index: int = mean_index
        self.std_index: int = std_index
        self.value_index: int = value_index

    def __call__(self, outputs: List[torch.Tensor], labels: List[np.ndarray],
                 weights: List[np.ndarray]):
        import torch.distributions as dist
        mean: torch.Tensor = outputs[self.mean_index]
        std: torch.Tensor = outputs[self.std_index]
        value: torch.Tensor = outputs[self.value_index]
        reward, advantage = weights
        action: torch.Tensor = torch.from_numpy(labels[0])
        distrib = dist.Normal(torch.tensor(mean), torch.tensor(std))
        reduce_axes: List[int] = list(range(1, len(action.shape)))
        log_prob: torch.Tensor = torch.sum(distrib.log_prob(action),
                                           reduce_axes)
        policy_loss: torch.Tensor = -torch.mean(
            torch.from_numpy(advantage) * log_prob)
        value_loss: torch.Tensor = torch.mean(
            torch.square(torch.from_numpy(reward) - value))
        entropy: torch.Tensor = torch.mean(distrib.entropy())
        return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
