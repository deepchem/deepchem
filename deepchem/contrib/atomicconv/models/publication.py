from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import time
import tempfile

import numpy as np
import tensorflow as tf

from deepchem.nn import model_ops
from deepchem.utils.save import log
from deepchem.models.tensorflow_models import TensorflowGraphModel
from deepchem.data import pad_features

from deepchem.contrib.atomicconv.acnn.atomicnet_ops import AtomicConvolutionLayer

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"


class AtomicConvModel(TensorflowGraphModel):
  def __init__(self,
               n_tasks,
               radial_params,
               atom_types,
               frag1_num_atoms,
               frag2_num_atoms,
               complex_num_atoms,
               max_num_neighbors,
               logdir=None,
               layer_sizes=[100],
               weight_init_stddevs=[0.1],
               bias_init_consts=[1.],
               penalty=0.0,
               penalty_type="l2",
               dropouts=[0.5],
               learning_rate=.001,
               momentum=.8,
               optimizer="adam",
               batch_size=48,
               conv_layers=1,
               boxsize=None,
               verbose=True,
               seed=None,
               sess=None):
      
    """Initialize TensorflowFragmentRegressor.

    Parameters
    ----------

    n_tasks: int
      Number of tasks.
    radial_params: list
      Of length l, where l is number of radial filters learned.
    atom_types: list
      Of length a, where a is number of atom_types for filtering.
    frag1_num_atoms: int
      Maximum number of atoms in fragment 1.
    frag2_num_atoms: int
      Maximum number of atoms in fragment 2.
    complex_num_atoms: int
      Maximum number of atoms in complex.
    max_num_neighbors: int
      Maximum number of neighbors per atom.
    logdir: str
      Path to model save directory.
    layer_sizes: list
      List of layer sizes.
    weight_init_stddevs: list
      List of standard deviations for weights (sampled from zero-mean
      gaussians). One for each layer.
    bias_init_consts: list
      List of bias initializations. One for each layer.
    penalty: float
      Amount of penalty (l2 or l1 applied)
    penalty_type: str
      Either "l2" or "l1"
    dropouts: list
      List of dropout amounts. One for each layer.
    learning_rate: float
      Learning rate for model.
    momentum: float
      Momentum. Only applied if optimizer=="momentum"
    optimizer: str
      Type of optimizer applied.
    batch_size: int
      Size of minibatches for training.
    conv_layers: int
      Number of atomic convolution layers (experimental feature).
    boxsize: float or None
      Simulation box length [Angstrom]. If None, no periodic boundary conditions.
    verbose: bool, optional (Default True)
      Whether to perform logging.
    seed: int, optional (Default None)
      If not none, is used as random seed for tensorflow.

    """

    self.n_tasks = n_tasks
    self.radial_params = radial_params
    self.atom_types = atom_types
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.complex_num_atoms = complex_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.conv_layers = conv_layers
    self.boxsize = boxsize

    self.layer_sizes = layer_sizes
    self.weight_init_stddevs = weight_init_stddevs
    self.bias_init_consts = bias_init_consts
    self.penalty = penalty
    self.penalty_type = penalty_type
    self.dropouts = dropouts
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.pad_batches = True
    self.verbose = verbose
    self.seed = seed
    
    self.sess = sess

    if logdir is not None:
      if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
      logdir = tempfile.mkdtemp()
    self.logdir = logdir

    # Path to save checkpoint files, which matches the
    # replicated supervisor's default path.
    self._save_path = os.path.join(logdir, 'model.ckpt')
    
    N = self.complex_num_atoms
    N_1 = self.frag1_num_atoms
    N_2 = self.frag2_num_atoms
    M = self.max_num_neighbors
    B = self.batch_size

    # Placeholders
    self.frag1_X_placeholder = tf.placeholder(tf.float32, shape=[B, N_1, 3])
    self.frag1_Nbrs_placeholder = tf.placeholder(tf.int32, shape=[B, N_1, M])
    self.frag1_Nbrs_Z_placeholder = tf.placeholder(tf.float32, shape=[B, N_1, M])
    self.frag2_X_placeholder = tf.placeholder(tf.float32, shape=[B, N_2, 3])
    self.frag2_Nbrs_placeholder = tf.placeholder(tf.int32, shape=[B, N_2, M])
    self.frag2_Nbrs_Z_placeholder = tf.placeholder(tf.float32, shape=[B, N_2, M])
    self.complex_X_placeholder = tf.placeholder(tf.float32, shape=[B, N, 3])
    self.complex_Nbrs_placeholder = tf.placeholder(tf.int32, shape=[B, N, M])
    self.complex_Nbrs_Z_placeholder = tf.placeholder(tf.float32, shape=[B, N, M])
    
    # Train graph
    self.output = self.build(training=True)
    self.label_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
    cost = 0.5 * tf.square(self.output - self.label_placeholder)
    
    self.loss = tf.reduce_mean(cost)
    
    if self.penalty != 0.0:
        print(("WARNING: Weight decay is actually broken in DC since it")
        (" can't handle single-valued variables, which we have."))
        #_penalty = model_ops.weight_decay(self.penalty_type, self.penalty)
        #self.loss += _penalty
    
    self.optimizer = model_ops.optimizer(self.optimizer, self.learning_rate,
                            self.momentum).minimize(self.loss)
                            
    model_var_len = len(tf.trainable_variables())


    # Eval graph
    self.output_eval = self.build(training=False)
    
    # Ensures total variable reuse in evaluation graph and that nothing was
    # inadvertantly added.
    assert model_var_len == len(tf.trainable_variables())
    
    #for v in tf.trainable_variables():
    #    print(v.name)

  def predict_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: dc.data.Dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
        labels: True labels.
        weights: Example weights.
      Note that the output and labels arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    
    len_unpadded = len(X)
    if self.pad_batches:
      X = pad_features(self.batch_size, X)

    # run eval data through the model
    n_tasks = self.n_tasks
    outputs = []

    n_samples = len(X)
    feed_dict = self.construct_feed_dict(X)

    data = self.sess.run(
        self.output_eval, feed_dict=feed_dict)

    batch_outputs = np.asarray(data[:n_tasks], dtype=float)
    # reshape to batch_size x n_tasks x ...
    if batch_outputs.ndim == 3:
      batch_outputs = batch_outputs.transpose((1, 0, 2))
    elif batch_outputs.ndim == 2:
      batch_outputs = batch_outputs.transpose((1, 0))
    # Handle edge case when batch-size is 1.
    elif batch_outputs.ndim == 1:
      n_samples = len(X)
      batch_outputs = batch_outputs.reshape((n_samples, n_tasks))
    else:
      raise ValueError('Unrecognized rank combination for output: %s' %
                       (batch_outputs.shape))
    # Prune away any padding that was added
    batch_outputs = batch_outputs[:n_samples]
    outputs.append(batch_outputs)

    outputs = np.squeeze(np.concatenate(outputs))

    outputs = np.copy(outputs)

    # Handle case of 0-dimensional scalar output
    if len(outputs.shape) > 0:
      return outputs[:len_unpadded]
    else:
      outputs = np.reshape(outputs, (1,))
      return outputs

  def construct_feed_dict(self, F_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    B = batch_size, N = max_num_atoms

    Parameters
    ----------

    F_b: np.ndarray of B tuples of (X_1, L_1, Z_1, X_2, L_2, Z_2, X, L, Z) 
      X_1: ndarray shape (N, 3).
        Fragment 1 Cartesian coordinates [Angstrom].
      L_1: dict with N keys.
        Fragment 1 neighbor list.
      Z_1: ndarray shape (N,).
        Fragment 1 atomic numbers.
      X_2: ndarray shape (N, 3).
        Fragment 2 Cartesian coordinates [Angstrom].
      L_2: dict with N keys.
        Fragment 2 neighbor list.
      Z_2: ndarray shape (N,).
        Fragment 2 atomic numbers.
      X: ndarray shape (N, 3).
        Complex Cartesian coordinates [Angstrom].
      L: dict with N keys.
        Complex neighbor list.
      Z: ndarray shape (N,).
        Complex atomic numbers.
    y_b: np.ndarray of shape (B, num_tasks)
      Tasks.
    w_b: np.ndarray of shape (B, num_tasks)
      Task weights.
    ids_b: List of length (B,) 
      Datapoint identifiers. Not currently used.

    Returns
    -------

    retval: dict
      Tensorflow feed dict

    """
    #start = arrow.now()
    N = self.complex_num_atoms
    N_1 = self.frag1_num_atoms
    N_2 = self.frag2_num_atoms
    M = self.max_num_neighbors
    
    batch_size = F_b.shape[0]
    num_features = F_b[0][0].shape[1]
    
    padded_data = []
    
    def pad(x, _N):
        pad_shape = [(0, _N - x.shape[0])]
        if len(x.shape) == 2:
            pad_shape.append((0, num_features - x.shape[1]))

        return np.lib.pad(x, pad_shape, 'constant', constant_values=0)
        
    frag1_X_b = []
    frag1_Z_b = []
    frag2_X_b = []
    frag2_Z_b = []
    complex_X_b = []
    complex_Z_b = []
    
    for example in F_b:
        frag1_X_b.append(pad(example[0], N_1))
        frag1_Z_b.append(pad(example[2], N_1))
        frag2_X_b.append(pad(example[3], N_2))
        frag2_Z_b.append(pad(example[5], N_2))
        complex_X_b.append(pad(example[6], N))
        complex_Z_b.append(pad(example[8], N))
            
    def get_nbrs(_F, _Z, _N, _M):
        nbrs = np.zeros((batch_size, _N, _M))
        nbrs_z = np.zeros((batch_size, _N, _M))
        for atom in range(_N):
          for i in range(batch_size):
            atom_nbrs = _F[i].get(atom, "")
            nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              nbrs_z[i, atom, j] = _Z[i, atom_j]
        return nbrs, nbrs_z
    
    frag1_Nbrs, frag1_Nbrs_Z = get_nbrs([x[1] for x in F_b], np.array(frag1_Z_b), N_1, M)
    frag2_Nbrs, frag2_Nbrs_Z = get_nbrs([x[4] for x in F_b], np.array(frag2_Z_b), N_2, M)
    complex_Nbrs, complex_Nbrs_Z = get_nbrs([x[7] for x in F_b], np.array(complex_Z_b), N, M)


    feed_dict = {
        self.frag1_X_placeholder: np.array(frag1_X_b),
        self.frag2_X_placeholder: np.array(frag2_X_b),
        self.complex_X_placeholder: np.array(complex_X_b),
        self.frag1_Nbrs_placeholder: frag1_Nbrs,
        self.frag1_Nbrs_Z_placeholder: frag1_Nbrs_Z,
        self.frag2_Nbrs_placeholder: frag2_Nbrs,
        self.frag2_Nbrs_Z_placeholder: frag2_Nbrs_Z,
        self.complex_Nbrs_placeholder: complex_Nbrs,
        self.complex_Nbrs_Z_placeholder: complex_Nbrs_Z,
        }
    
    if y_b is not None:
        feed_dict[self.label_placeholder] = y_b
    #print(arrow.now() - start)
    return feed_dict

  def build(self, training):
  
      if not training:
          tf.get_variable_scope().reuse_variables()
      
      N = self.complex_num_atoms
      N_1 = self.frag1_num_atoms
      N_2 = self.frag2_num_atoms
      M = self.max_num_neighbors
      B = self.batch_size
  
      layer_sizes = self.layer_sizes
      weight_init_stddevs = self.weight_init_stddevs
      bias_init_consts = self.bias_init_consts
      dropouts = self.dropouts
      boxsize = self.boxsize
      conv_layers = self.conv_layers
      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
      }
      
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      num_layers = lengths_set.pop()
      assert num_layers > 0, 'Must have some layers defined.'
      radial_params = self.radial_params
      atom_types = self.atom_types
      
      with tf.variable_scope('frag1'):
        frag1_layer = AtomicConvolutionLayer(
          self.frag1_X_placeholder, self.frag1_Nbrs_placeholder,
          self.frag1_Nbrs_Z_placeholder, atom_types, radial_params, boxsize, B,
          N_1, M, 3)
          
      with tf.variable_scope('frag2'):
        frag2_layer = AtomicConvolutionLayer(
          self.frag2_X_placeholder, self.frag2_Nbrs_placeholder,
          self.frag2_Nbrs_Z_placeholder, atom_types, radial_params, boxsize, B,
          N_2, M, 3)
          
      with tf.variable_scope('complex'):
        complex_layer = AtomicConvolutionLayer(
          self.complex_X_placeholder, self.complex_Nbrs_placeholder,
          self.complex_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
          B, N, M, 3)
          
      def compute_frag(frag_layer, nbr_placeholder, nbr_z_placeholder, _N):
        for x in range(conv_layers - 1):
            frag_layer = tf.transpose(frag_layer, [1, 2, 0])
            l = int(frag_layer.get_shape()[-1])
            frag_layer = AtomicConvolutionLayer(
                frag_layer, nbr_placeholder,
                nbr_z_placeholder, atom_types, radial_params, boxsize,
                B, _N, M, l)
        return tf.expand_dims(tf.transpose(frag_layer, [1, 2, 0]), 1)
          

      frag1_layer = compute_frag(frag1_layer, self.frag1_Nbrs_placeholder, self.frag1_Nbrs_Z_placeholder, N_1)
      frag2_layer = compute_frag(frag2_layer, self.frag2_Nbrs_placeholder, self.frag2_Nbrs_Z_placeholder, N_2)
      complex_layer = compute_frag(complex_layer, self.complex_Nbrs_placeholder, self.complex_Nbrs_Z_placeholder, N)

      output = []

      def convolve(frag_layer, reuse=None):
        for i, (s, w, b) in enumerate(zip(layer_sizes, weight_init_stddevs, bias_init_consts)):
            if i != len(layer_sizes) - 1:
                act = tf.nn.relu
            else:
                act = None
            # TODO: layer = model_ops.dropout(layer, dropouts[i], training)
            frag_layer = tf.contrib.slim.convolution2d(frag_layer,
                        layer_sizes[i],
                        kernel_size=1,
                        stride=1,
                        padding='VALID',
                        activation_fn=act,

                        weights_initializer=tf.truncated_normal_initializer(stddev=w),
                        biases_initializer=tf.constant_initializer(b),
                        reuse=reuse,
                        scope='conv{}'.format(i))
        return tf.reduce_sum(frag_layer, [1, 2, 3])
      

      frag1_energy = convolve(frag1_layer)
      frag2_energy = convolve(frag2_layer, True)
      complex_energy = convolve(complex_layer, True)
      
      binding_energy = complex_energy - frag1_energy - frag2_energy
      output.append(binding_energy)

      return output


  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          sess=None,
          **kwargs):
    """Fit the model.

    Parameters
    ---------- 
    dataset: dc.data.Dataset
      Dataset object holding training data 
    nb_epoch: 10
      Number of training epochs.
    max_checkpoints_to_keep: int
      Maximum number of checkpoints to keep; older checkpoints will be deleted.
    log_every_N_batches: int
      Report every N batches. Useful for training on very large datasets,
      where epochs can take long time to finish.

    Raises
    ------
    AssertionError
      If model is not in training mode.
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    log("Training for %d epochs" % nb_epoch, self.verbose)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
    # Save an initial checkpoint.
    #saver.save(sess, self._save_path, global_step=0)
    for epoch in range(nb_epoch):
      losses = []
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          # Turns out there are valid cases where we don't want pad-batches
          # on by default.
          #dataset.iterbatches(batch_size, pad_batches=True)):
          dataset.iterbatches(
              self.batch_size, pad_batches=self.pad_batches)):
        if ind % log_every_N_batches == 0 and ind != 0:
          log("On batch %d" % ind, self.verbose)
        # Run training op.
        feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)

        fetches = [self.optimizer, self.loss, self.output]
        _, loss, output = sess.run(fetches, feed_dict=feed_dict)
        losses.append(loss)

      # Trains fast enough so let's not save intermediate steps for now
      #saver.save(sess, self._save_path, global_step=epoch)
      
      avg_loss = np.mean(losses)
      log('Ending epoch %d: Average loss %g' % (epoch, avg_loss),
          self.verbose)
  
    # Always save a final checkpoint when complete.
    saver.save(sess, self._save_path, global_step=epoch + 1)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING



