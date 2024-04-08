import pytest
import numpy as np
try:
    import torch
    from deepchem.rl.torch_rl import A2CLossDiscrete, A2CLossContinuous
    has_pytorch = True
except:
    has_pytorch = False


@pytest.mark.torch
def test_A2CLossDiscrete():
    outputs = [
        torch.tensor([[
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.2
        ]]),
        torch.tensor([0.], requires_grad=True)
    ]
    labels = np.array([[
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.
    ]],
                      dtype=np.float32)
    discount = np.array([
        -1.0203744, -0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097,
        -0.9901, 0.01, -1., 0.
    ],
                        dtype=np.float32)
    advantage = np.array([
        -1.0203744, -0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097,
        -0.9901, 0.01, -1., 0.
    ],
                         dtype=np.float32)
    loss = A2CLossDiscrete(value_weight=1.0,
                           entropy_weight=0.01,
                           action_prob_index=0,
                           value_index=1)
    loss_val = loss(outputs, [labels], [discount, advantage])
    assert round(loss_val.item(), 4) == 1.2541


@pytest.mark.torch
def test_A2CLossContinuous():
    outputs = [
        torch.tensor(
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            dtype=torch.float32,
            requires_grad=True),
        torch.tensor([10.], dtype=torch.float32, requires_grad=True),
        torch.tensor([[27.717865], [28.860144]],
                     dtype=torch.float32,
                     requires_grad=True)
    ]
    labels = np.array(
        [[-4.897339], [3.4308329], [-4.527725], [-7.3000813], [-1.9869075],
         [20.664988], [-8.448957], [10.580486], [10.017258], [17.884674]],
        dtype=np.float32)
    discount = np.array([
        4.897339, -8.328172, 7.958559, 2.772356, -5.313174, -22.651896,
        29.113945, -19.029444, 0.56322646, -7.867417
    ],
                        dtype=np.float32)
    advantage = np.array([
        -5.681633, -20.57494, -1.4520378, -9.348538, -18.378199, -33.907513,
        25.572464, -32.485718, -6.412546, -15.034998
    ],
                         dtype=np.float32)
    loss = A2CLossContinuous(value_weight=1.0,
                             entropy_weight=0.01,
                             mean_index=0,
                             std_index=1,
                             value_index=2)
    loss_val = loss(outputs, [labels], [discount, advantage])
    assert round(loss_val.item(), 4) == 1050.2310
