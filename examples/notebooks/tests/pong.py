def test_notebook():
  
  # coding: utf-8
  
  # # Pong in DeepChem with A3C
  # This notebook demonstrates using reinforcement learning to train an agent to play Pong.
  # 
  # The first step is to create an `Environment` that implements this task.  Fortunately,
  # OpenAI Gym already provides an implementation of Pong (and many other tasks appropriate
  # for reinforcement learning).  DeepChem's `GymEnvironment` class provides an easy way to
  # use environments from OpenAI Gym.  We could just use it directly, but in this case we
  # subclass it and preprocess the screen image a little bit to make learning easier.
  
  # In[ ]:
  
  
  import deepchem as dc
  import numpy as np
  
  class PongEnv(dc.rl.GymEnvironment):
    def __init__(self):
      super(PongEnv, self).__init__('Pong-v0')
      self._state_shape = (80, 80)
    
    @property
    def state(self):
      # Crop everything outside the play area, reduce the image size,
      # and convert it to black and white.
      cropped = np.array(self._state)[34:194, :, :]
      reduced = cropped[0:-1:2, 0:-1:2]
      grayscale = np.sum(reduced, axis=2)
      bw = np.zeros(grayscale.shape)
      bw[grayscale != 233] = 1
      return bw
  
    def __deepcopy__(self, memo):
      return PongEnv()
  
  env = PongEnv()
  
  
  # Next we create a network to implement the policy.  We begin with two convolutional layers to process
  # the image.  That is followed by a dense (fully connected) layer to provide plenty of capacity for game
  # logic.  We also add a small Gated Recurrent Unit.  That gives the network a little bit of memory, so
  # it can keep track of which way the ball is moving.
  # 
  # We concatenate the dense and GRU outputs together, and use them as inputs to two final layers that serve as the
  # network's outputs.  One computes the action probabilities, and the other computes an estimate of the
  # state value function.
  
  # In[ ]:
  
  
  import deepchem.models.tensorgraph.layers as layers
  import tensorflow as tf
  
  class PongPolicy(dc.rl.Policy):
      def create_layers(self, state, **kwargs):
          conv1 = layers.Conv2D(num_outputs=16, in_layers=state, kernel_size=8, stride=4)
          conv2 = layers.Conv2D(num_outputs=32, in_layers=conv1, kernel_size=4, stride=2)
          dense = layers.Dense(out_channels=256, in_layers=layers.Flatten(in_layers=conv2), activation_fn=tf.nn.relu)
          gru = layers.GRU(n_hidden=16, batch_size=1, in_layers=layers.Reshape(shape=(1, -1, 256), in_layers=dense))
          concat = layers.Concat(in_layers=[dense, layers.Reshape(shape=(-1, 16), in_layers=gru)])
          action_prob = layers.Dense(out_channels=env.n_actions, activation_fn=tf.nn.softmax, in_layers=concat)
          value = layers.Dense(out_channels=1, in_layers=concat)
          return {'action_prob':action_prob, 'value':value}
  
  policy = PongPolicy()
  
  
  # We will optimize the policy using the Asynchronous Advantage Actor Critic (A3C) algorithm.  There are lots of hyperparameters we could specify at this point, but the default values for most of them work well on this problem.  The only one we need to customize is the learning rate.
  
  # In[ ]:
  
  
  from deepchem.models.tensorgraph.optimizers import Adam
  a3c = dc.rl.A3C(env, policy, model_dir='model', optimizer=Adam(learning_rate=0.0002))
  
  
  # Optimize for as long as you have patience to.  By 1 million steps you should see clear signs of learning.  Around 3 million steps it should start to occasionally beat the game's built in AI.  By 7 million steps it should be winning almost every time.  Running on my laptop, training takes about 20 minutes for every million steps.
  
  # In[ ]:
  
  
  a3c.fit(1000)
  # Change this for how long you have the patience
  million = 1
  num_rounds = 0
  for round in range(num_rounds):
      a3c.fit(million, restore=True)
  
  
  # Let's watch it play and see how it does!
  
  # In[ ]:
  
  
  # from datetime import datetime
  # def render_env(env):
  #     try:
  #         env.env.render()
  #     except Exception as e:
  #         print(e)
  #
  # a3c.restore()
  # env.reset()
  # start = datetime.now()
  # while (datetime.now() - start).total_seconds() < 120:
  #     render_env(env)
  #     env.step(a3c.select_action(env.state))
  
