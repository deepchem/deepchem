import copy
import random
import shutil

import numpy as np
import tensorflow as tf

import deepchem as dc
import deepchem.rl.envs.tictactoe
from deepchem.models.tensorgraph.layers import Flatten, Dense, SoftMax, \
    BatchNorm, Squeeze
from deepchem.models.tensorgraph.optimizers import Adam


class TicTacToePolicy(dc.rl.Policy):

  def create_layers(self, state, **kwargs):
    d1 = Flatten(in_layers=state)
    d2 = Dense(
        in_layers=[d1],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=64)
    d3 = Dense(
        in_layers=[d2],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=32)
    d4 = Dense(
        in_layers=[d3],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=16)
    d4 = BatchNorm(in_layers=[d4])
    d5 = Dense(in_layers=[d4], activation_fn=None, out_channels=9)
    value = Dense(in_layers=[d4], activation_fn=None, out_channels=1)
    value = Squeeze(squeeze_dims=1, in_layers=[value])
    probs = SoftMax(in_layers=[d5])
    return {'action_prob': probs, 'value': value}


def eval_tic_tac_toe(value_weight,
                     num_epoch_rounds=1,
                     games=10**4,
                     rollouts=10**5):
  """
    Returns the average reward over 1k games after 100k rollouts
    :param value_weight:
    :return:
    """
  env = deepchem.rl.envs.tictactoe.TicTacToeEnvironment()
  policy = TicTacToePolicy()
  model_dir = "/tmp/tictactoe"
  try:
    shutil.rmtree(model_dir)
  except:
    pass

  avg_rewards = []
  for j in range(num_epoch_rounds):
    a3c = dc.rl.A3C(
        env,
        policy,
        entropy_weight=0.01,
        value_weight=value_weight,
        model_dir=model_dir,
        optimizer=Adam(learning_rate=0.001))
    try:
      a3c.restore()
    except:
      print("unable to restore")
      pass
    a3c.fit(rollouts)
    rewards = []
    for i in range(games):
      env.reset()
      reward = -float('inf')
      while not env._terminated:
        action = a3c.select_action(env._state)
        reward = env.step(action)
      rewards.append(reward)
    avg_rewards.append({(j + 1) * rollouts: np.mean(rewards)})
  return avg_rewards


def main():
  value_weight = 6.0
  score = eval_tic_tac_toe(value_weight, num_epoch_rounds=3)
  print(score)


if __name__ == "__main__":
  main()
