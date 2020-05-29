Reinforcement Learning
======================
Reinforcement Learning is a powerful technique for learning when you
have access to a simulator. That is, suppose that you have a high
fidelity way of predicting the outcome of an experiment. This is
perhaps a physics engine, perhaps a chemistry engine, or anything. And
you'd like to solve some task within this engine. You can use
reinforcement learning for this purpose.


Environments
------------

.. autoclass:: deepchem.rl.Environment
  :members:

.. autoclass:: deepchem.rl.GymEnvironment
  :members:

Policies
--------

.. autoclass:: deepchem.rl.Policy
  :members:

A2C
---

.. autoclass:: deepchem.rl.a2c.A2C
  :members:

.. autoclass:: deepchem.rl.a2c.A2CLossDiscrete
  :members:

PPO
---

.. autoclass:: deepchem.rl.ppo.PPO
  :members:

.. autoclass:: deepchem.rl.ppo.PPOLoss
  :members:
