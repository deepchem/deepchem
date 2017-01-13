"""
Copies Classes from keras to remove dependency.

Most of this code is copied over from Keras. Hoping to use as a staging
area while we remove our Keras dependency.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from . import initializations
from . import regularizers
from . import activations
from . import constraints
import tensorflow as tf
from keras import backend as K

def get_ndim(x):
  """Returns the number of axes in a tensor, as an integer.

  # Arguments
      x: Tensor or variable.

  # Returns
      Integer (scalar), number of axes.
  """
  dims = x.get_shape()._dims
  if dims is not None:
    return len(dims)
  return None

def dtype(x):
  """Returns the dtype of a Keras tensor or variable, as a string.

  # Arguments
      x: Tensor or variable.

  # Returns
      String, dtype of `x`.
  """
  return x.dtype.name

def to_list(x):
  """This normalizes a list/tensor into a list.

  If a tensor is passed, we return
  a list of size 1 containing the tensor.
  """
  if isinstance(x, list):
      return x
  return [x]

class InputSpec(object):
  """This specifies the ndim, dtype and shape of every input to a layer.
  Every layer should expose (if appropriate) an `input_spec` attribute:
  a list of instances of InputSpec (one per input tensor).

  A None entry in a shape is compatible with any dimension,
  a None shape is compatible with any shape.
  """

  def __init__(self, dtype=None, shape=None, ndim=None):
    if isinstance(ndim, str):
      if '+' not in ndim:
          raise ValueError('When passing a str "ndim", '
                           'it should have the form "2+", "3+", etc.')
      int_ndim = ndim[:ndim.find('+')]
      if not int_ndim.isdigit():
          raise ValueError('When passing a str "ndim", '
                           'it should have the form "2+", "3+", etc.')
    if shape is not None:
      self.ndim = len(shape)
    else:
      self.ndim = ndim
    self.dtype = dtype
    self.shape = shape

class Node(object):
  """A `Node` describes the connectivity between two layers.

  Each time a layer is connected to some new input,
  a node is added to `layer.inbound_nodes`.
  Each time the output of a layer is used by another layer,
  a node is added to `layer.outbound_nodes`.

  # Attributes
    outbound_layer: the layer that takes
        `input_tensors` and turns them into `output_tensors`.
    inbound_layers: a list of layers, the same length as `input_tensors`,
        the layers from where `input_tensors` originate.
    node_indices: a list of integers, the same length as `inbound_layers`.
        `node_indices[i]` is the origin node of `input_tensors[i]`
        (necessary since each inbound layer might have several nodes,
        e.g. if the layer is being shared with a different data stream).
    tensor_indices: a list of integers,
        the same length as `inbound_layers`.
        `tensor_indices[i]` is the index of `input_tensors[i]` within the
        output of the inbound layer
        (necessary since each inbound layer might
        have multiple tensor outputs, with each one being
        independently manipulable).
    input_tensors: list of input tensors.
    output_tensors: list of output tensors.
    input_masks: list of input masks (a mask can be a tensor, or None).
    output_masks: list of output masks (a mask can be a tensor, or None).
    input_shapes: list of input shape tuples.
    output_shapes: list of output shape tuples.

  `node_indices` and `tensor_indices` are basically fine-grained coordinates
  describing the origin of the `input_tensors`, verifying the following:

  `input_tensors[i] == inbound_layers[i].inbound_nodes[node_indices[i]].output_tensors[tensor_indices[i]]`

  A node from layer A to layer B is added to:
      A.outbound_nodes
      B.inbound_nodes
  """

  def __init__(self, outbound_layer,
               inbound_layers, node_indices, tensor_indices,
               input_tensors, output_tensors,
               input_masks, output_masks,
               input_shapes, output_shapes):
    # Layer instance (NOT a list).
    # this is the layer that takes a list of input tensors
    # and turns them into a list of output tensors.
    # the current node will be added to
    # the inbound_nodes of outbound_layer.
    self.outbound_layer = outbound_layer

    # The following 3 properties describe where
    # the input tensors come from: which layers,
    # and for each layer, which node and which
    # tensor output of each node.

    # List of layer instances
    self.inbound_layers = inbound_layers  
    # List of integers, 1:1 mapping with inbound_layers.
    self.node_indices = node_indices  
    # List of integers, 1:1 mapping with inbound_layers.
    self.tensor_indices = tensor_indices  

    # Tensor inputs and outputs of outbound_layer.
    # List of tensors. 1:1 mapping with inbound_layers.
    self.input_tensors = input_tensors  
    # List of tensors, created by outbound_layer.call().
    self.output_tensors = output_tensors  

    # input and output masks
    # List of tensors, 1:1 mapping with input_tensor.
    self.input_masks = input_masks  
    # List of tensors, created by outbound_layer.compute_mask().
    self.output_masks = output_masks  

    # input and output shapes
    # List of shape tuples, shapes of input_tensors.
    self.input_shapes = input_shapes  
    # List of shape tuples, shapes of output_tensors.
    self.output_shapes = output_shapes  

    # Add nodes to all layers involved.
    for layer in inbound_layers:
      if layer is not None:
        layer.outbound_nodes.append(self)
    outbound_layer.inbound_nodes.append(self)

  @classmethod
  def create_node(cls, outbound_layer, inbound_layers, node_indices=None,
                  tensor_indices=None):
    if not node_indices:
        node_indices = [0 for _ in range(len(inbound_layers))]
    else:
        assert len(node_indices) == len(inbound_layers)
    if not tensor_indices:
        tensor_indices = [0 for _ in range(len(inbound_layers))]

    input_tensors = []
    input_masks = []
    input_shapes = []

    for inbound_layer, node_index, tensor_index in zip(
        inbound_layers, node_indices, tensor_indices):
      inbound_node = inbound_layer.inbound_nodes[node_index]
      input_tensors.append(inbound_node.output_tensors[tensor_index])
      input_masks.append(inbound_node.output_masks[tensor_index])
      input_shapes.append(inbound_node.output_shapes[tensor_index])

    assert len(input_shapes) == len(input_tensors) == len(input_masks)

    if len(input_tensors) == 1:
      output_tensors = to_list(outbound_layer.call(
          input_tensors[0], mask=input_masks[0]))
      output_masks = to_list(outbound_layer.compute_mask(
          input_tensors[0], input_masks[0]))
      # TODO: try to auto-infer shape
      # if exception is raised by get_output_shape_for.
      output_shapes = to_list(outbound_layer.get_output_shape_for(input_shapes[0]))
    else:
      output_tensors = to_list(outbound_layer.call(input_tensors, mask=input_masks))
      output_masks = to_list(outbound_layer.compute_mask(input_tensors, input_masks))
      output_shapes = to_list(outbound_layer.get_output_shape_for(input_shapes))

    if not output_tensors or output_tensors[0] is None:
      raise TypeError('The `call` method of layer "' +
                      outbound_layer.name +
                      '" should return a tensor. Found: ' +
                      str(output_tensors[0]))
    if len(output_tensors) != len(output_shapes):
      raise ValueError('The `get_output_shape_for` method of layer "' +
                       outbound_layer.name +
                       '"" should return one shape tuple per '
                       'output tensor of the layer. Found: ' +
                       str(output_shapes))
    if len(output_tensors) != len(output_masks):
      raise ValueError('The `compute_mask` method of layer "' +
                       outbound_layer.name +
                       '" should return one mask tensor per '
                       'output tensor of the layer. Found: ' +
                       str(output_masks))

    for i in range(len(output_tensors)):
      output_tensors[i]._keras_shape = output_shapes[i]
      output_tensors[i]._uses_learning_phase = (
          any([x._uses_learning_phase for x in input_tensors])
          or outbound_layer.uses_learning_phase)
      output_tensors[i]._keras_history = (
          outbound_layer, len(outbound_layer.inbound_nodes), i)

    return cls(outbound_layer,
               inbound_layers, node_indices, tensor_indices,
               input_tensors, output_tensors,
               input_masks, output_masks,
               input_shapes, output_shapes)

class Layer(object):
  """Abstract base layer class.

  # Properties
    name: String, must be unique within a model.
    input_spec: List of InputSpec class instances
        each entry describes one required input:
            - ndim
            - dtype
        A layer with `n` input tensors must have
        an `input_spec` of length `n`.
    trainable: Boolean, whether the layer weights
        will be updated during training.
    uses_learning_phase: Whether any operation
        of the layer uses `K.in_training_phase()`
        or `K.in_test_phase()`.
    input_shape: Shape tuple. Provided for convenience,
        but note that there may be cases in which this
        attribute is ill-defined (e.g. a shared layer
        with multiple input shapes), in which case
        requesting `input_shape` will raise an Exception.
        Prefer using `layer.get_input_shape_for(input_shape)`,
        or `layer.get_input_shape_at(node_index)`.
    output_shape: Shape tuple. See above.
    inbound_nodes: List of nodes.
    outbound_nodes: List of nodes.
    supports_masking: Boolean.
    input, output: Input/output tensor(s). Note that if the layer is used
        more than once (shared layer), this is ill-defined
        and will raise an exception. In such cases, use
        `layer.get_input_at(node_index)`.
    input_mask, output_mask: Same as above, for masks.
    trainable_weights: List of variables.
    non_trainable_weights: List of variables.
    weights: The concatenation of the lists trainable_weights and
        non_trainable_weights (in this order).
    constraints: Dict mapping weights to constraints.

  # Methods
    call(x, mask=None): Where the layer's logic lives.
    __call__(x, mask=None): Wrapper around the layer logic (`call`).
        If x is a Keras tensor:
            - Connect current layer with last layer from tensor:
                `self.add_inbound_node(last_layer)`
            - Add layer to tensor history
        If layer is not built:
            - Build from x._keras_shape
    get_weights()
    set_weights(weights)
    count_params()
    get_output_shape_for(input_shape)
    compute_mask(x, mask)
    get_input_at(node_index)
    get_output_at(node_index)
    get_input_shape_at(node_index)
    get_output_shape_at(node_index)
    get_input_mask_at(node_index)
    get_output_mask_at(node_index)

  # Internal methods:
    build(input_shape)
    add_inbound_node(layer, index=0)
    create_input_layer()
  """

  def __init__(self, **kwargs):
    # These properties should have been set
    # by the child class, as appropriate.
    if not hasattr(self, 'input_spec'):
      self.input_spec = None
    if not hasattr(self, 'supports_masking'):
      self.supports_masking = False
    if not hasattr(self, 'uses_learning_phase'):
      self.uses_learning_phase = False

    # These lists will be filled via successive calls
    # to self.add_inbound_node().
    self.inbound_nodes = []
    self.outbound_nodes = []

    # These properties will be set upon call of self.build(),
    # which itself will be called upon self.add_inbound_node if necessary.
    if not hasattr(self, '_trainable_weights'):
      self._trainable_weights = []
    if not hasattr(self, '_non_trainable_weights'):
      self._non_trainable_weights = []
    if not hasattr(self, 'losses'):
      self.losses = []
    if not hasattr(self, 'constraints'):
      self.constraints = {}  # dict {tensor: constraint instance}
    self.built = False

    # These properties should be set by the user via keyword arguments.
    # note that 'input_dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {'input_shape',
                      'batch_input_shape',
                      'input_dtype',
                      'name',
                      'trainable'}
    for kwarg in kwargs.keys():
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)
    name = kwargs.get('name')
    if not name:
      prefix = self.__class__.__name__.lower()
      name = prefix + '_' + str(K.get_uid(prefix))
    self.name = name

    self.trainable = kwargs.get('trainable', True)
    if 'batch_input_shape' in kwargs or 'input_shape' in kwargs:
      # In this case we will create an input layer
      # to insert before the current layer
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        batch_input_shape = (None,) + tuple(kwargs['input_shape'])
      self.batch_input_shape = batch_input_shape
      input_dtype = kwargs.get('input_dtype', tf.float32)
      self.input_dtype = input_dtype

  @property
  def trainable_weights(self):
    trainable = getattr(self, 'trainable', True)
    if trainable:
      return self._trainable_weights
    else:
      return []

  @trainable_weights.setter
  def trainable_weights(self, weights):
    self._trainable_weights = weights

  @property
  def non_trainable_weights(self):
    trainable = getattr(self, 'trainable', True)
    if not trainable:
      return self._trainable_weights + self._non_trainable_weights
    else:
      return self._non_trainable_weights

  @non_trainable_weights.setter
  def non_trainable_weights(self, weights):
    self._non_trainable_weights = weights

  def create_input_layer(self, batch_input_shape,
                         input_dtype=None, name=None):
    if not name:
      prefix = self.__class__.__name__.lower() + '_input_'
      name = prefix + str(K.get_uid(prefix))
    if not input_dtype:
      input_dtype = tf.float32

    self.batch_input_shape = batch_input_shape
    self.input_dtype = input_dtype

    # Instantiate the input layer.
    x = Input(batch_shape=batch_input_shape,
              dtype=input_dtype, name=name)
    # This will build the current layer
    # and create the node connecting the current layer
    # to the input layer we just created.
    self(x)

  def add_weight(self, shape, initializer, name=None,
                 trainable=True,
                 regularizer=None,
                 constraint=None):
    """Adds a weight variable to the layer.

    # Arguments
        shape: The shape tuple of the weight.
        initializer: An Initializer instance (callable).
        trainable: A boolean, whether the weight should
            be trained via backprop or not (assuming
            that the layer itself is also trainable).
        regularizer: An optional Regularizer instance.
    """
    initializer = initializations.get(initializer)
    weight = initializer(shape, name=name)
    if regularizer is not None:
      self.add_loss(regularizer(weight))
    if constraint is not None:
      self.constraints[weight] = constraint
    if trainable:
      self._trainable_weights.append(weight)
    else:
      self._non_trainable_weights.append(weight)
    return weight

  def call(self, x, mask=None):
    """This is where the layer's logic lives.

    # Arguments
        x: input tensor, or list/tuple of input tensors.
        mask: a masking tensor (or list of tensors). Used mainly in RNNs.

    # Returns:
        A tensor or list/tuple of tensors.
    """
    return x

  def __call__(self, x, mask=None):
    """Wrapper around self.call(), for handling
    internal Keras references.

    If a Keras tensor is passed:
      - We call self.add_inbound_node().
      - If necessary, we `build` the layer to match
          the _keras_shape of the input(s).
      - We update the _keras_shape of every input tensor with
          its new shape (obtained via self.get_output_shape_for).
          This is done as part of add_inbound_node().
      - We update the _keras_history of the output tensor(s)
          with the current layer.
          This is done as part of add_inbound_node().

    # Arguments
      x: Can be a tensor or list/tuple of tensors.
      mask: Tensor or list/tuple of tensors.
    """
    if not self.built:
      # Collect input shapes to build layer.
      input_shapes = []
      for x_elem in to_list(x):
        if hasattr(x_elem, '_keras_shape'):
          input_shapes.append(x_elem._keras_shape)
        elif hasattr(K, 'int_shape'):
          input_shapes.append(K.int_shape(x_elem))
        else:
          raise ValueError('You tried to call layer "' + self.name +
                           '". This layer has no information'
                           ' about its expected input shape, '
                           'and thus cannot be built. '
                           'You can build it manually via: '
                           '`layer.build(batch_input_shape)`')
      if len(input_shapes) == 1:
        self.build(input_shapes[0])
      else:
        self.build(input_shapes)
      self.built = True

    input_tensors = to_list(x)
    inbound_layers = []
    node_indices = []
    tensor_indices = []
    for input_tensor in input_tensors:
      if hasattr(input_tensor, '_keras_history') and input_tensor._keras_history:
        # This is a Keras tensor.
        previous_layer, node_index, tensor_index = input_tensor._keras_history
        inbound_layers.append(previous_layer)
        node_indices.append(node_index)
        tensor_indices.append(tensor_index)
      else:
        inbound_layers = None
        break

    if inbound_layers:
      # This will call layer.build() if necessary.
      self.add_inbound_node(inbound_layers, node_indices, tensor_indices)
      # Outputs were already computed when calling self.add_inbound_node.
      outputs = self.inbound_nodes[-1].output_tensors
    else:
      # This case appears if the input was not a Keras tensor.
      outputs = to_list(self.call(x, mask))

    # Apply activity regularizer if any:
    if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
      regularization_losses = [self.activity_regularizer(x) for x in outputs]
      self.add_loss(regularization_losses, input_tensors)

    # If single output tensor: return it,
    # else return a list (at least 2 elements).
    if len(outputs) == 1:
      return outputs[0]
    else:
      return outputs

  def add_inbound_node(self, inbound_layers,
                       node_indices=None, tensor_indices=None):
    """
    # Arguments
      inbound_layers: Can be a layer instance
          or a list/tuple of layer instances.
      node_indices: Integer (or list of integers).
          The input layer might have a number of
          parallel output streams;
          this is the index of the stream (in the input layer)
          where to connect the current layer.
      tensor_indices: Integer or list of integers.
          The output of the inbound node might be a list/tuple
          of tensor, and we might only be interested in
          one specific entry.
          This index allows you to specify the index of
          the entry in the output list
          (if applicable). "None" means that we take all outputs
          (as a list).
    """
    inbound_layers = to_list(inbound_layers)
    if not node_indices:
      node_indices = [0 for _ in range(len(inbound_layers))]
    else:
      node_indices = to_list(node_indices)
      assert len(node_indices) == len(inbound_layers)
    if not tensor_indices:
      tensor_indices = [0 for _ in range(len(inbound_layers))]
    else:
      tensor_indices = to_list(tensor_indices)

    if not self.built:
      # collect input_shapes for call to build()
      input_shapes = []
      for layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
          input_shapes.append(layer.inbound_nodes[node_index].output_shapes[tensor_index])
      # call build()
      if len(input_shapes) == 1:
          self.build(input_shape=input_shapes[0])
      else:
          self.build(input_shape=input_shapes)
      self.built = True
    # creating the node automatically updates self.inbound_nodes
    # as well as outbound_nodes on inbound layers.
    Node.create_node(self, inbound_layers, node_indices, tensor_indices)

  def get_output_shape_for(self, input_shape):
    """Computes the output shape of the layer given
    an input shape (assumes that the layer will be built
    to match that input shape).

    # Arguments
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.
    """
    return input_shape

  def compute_mask(self, input, input_mask=None):
    """Computes an output masking tensor, given an input tensor
    (or list thereof) and an input mask (or list thereof).

    # Arguments
        input: Tensor or list of tensors.
        input_mask: Tensor or list of tensors.

    # Returns
        None or a tensor (or list of tensors,
            one per output tensor of the layer).
    """
    if not hasattr(self, 'supports_masking') or not self.supports_masking:
      if input_mask is not None:
        if isinstance(input_mask, list):
          if any(input_mask):
            raise ValueError('Layer ' + self.name +
                             ' does not support masking, '
                             'but was passed an input_mask: ' +
                             str(input_mask))
        else:
          raise ValueError('Layer ' + self.name +
                           ' does not support masking, '
                           'but was passed an input_mask: ' +
                           str(input_mask))
      # masking not explicitly supported: return None as mask
      return None
    # if masking is explictly supported, by default
    # carry over the input mask
    return input_mask

  def build(self, input_shape):
    """Creates the layer weights.
    Must be implemented on all layers that have weights.

    # Arguments
      input_shape: Keras tensor (future input to layer)
        or list/tuple of Keras tensors to reference
        for weight shape computations.
    """
    self.built = True

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    """Retrieves an attribute (e.g. input_tensors) from a node.

    # Arguments
        node_index: Integer index of the node from which
            to retrieve the attribute.
        attr: Exact node attribute name.
        attr_name: Human-readable attribute name, for error messages.
    """
    if not self.inbound_nodes:
      raise RuntimeError('The layer has never been called '
                         'and thus has no defined ' + attr_name + '.')
    if not len(self.inbound_nodes) > node_index:
      raise ValueError('Asked to get ' + attr_name +
                       ' at node ' + str(node_index) +
                       ', but the layer has only ' +
                       str(len(self.inbound_nodes)) + ' inbound nodes.')
    values = getattr(self.inbound_nodes[node_index], attr)
    if len(values) == 1:
      return values[0]
    else:
      return values

  def get_input_shape_at(self, node_index):
    """Retrieves the input shape(s) of a layer at a given node.
    """
    return self._get_node_attribute_at_index(node_index,
                                             'input_shapes',
                                             'input shape')

  def get_output_shape_at(self, node_index):
    """Retrieves the output shape(s) of a layer at a given node.
    """
    return self._get_node_attribute_at_index(node_index,
                                             'output_shapes',
                                             'output shape')

  def get_input_at(self, node_index):
    """Retrieves the input tensor(s) of a layer at a given node.
    """
    return self._get_node_attribute_at_index(node_index,
                                             'input_tensors',
                                             'input')

  def get_output_at(self, node_index):
    """Retrieves the output tensor(s) of a layer at a given node.
    """
    return self._get_node_attribute_at_index(node_index,
                                             'output_tensors',
                                             'output')

  def get_input_mask_at(self, node_index):
    """Retrieves the input mask tensor(s) of a layer at a given node.
    """
    return self._get_node_attribute_at_index(node_index,
                                             'input_masks',
                                             'input mask')

  def get_output_mask_at(self, node_index):
    """Retrieves the output mask tensor(s) of a layer at a given node.
    """
    return self._get_node_attribute_at_index(node_index,
                                             'output_masks',
                                             'output mask')

  @property
  def input(self):
    """Retrieves the input tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    """
    if len(self.inbound_nodes) > 1:
      raise AttributeError('Layer ' + self.name +
                           ' has multiple inbound nodes, '
                           'hence the notion of "layer input" '
                           'is ill-defined. '
                           'Use `get_input_at(node_index)` instead.')
    elif not self.inbound_nodes:
      raise AttributeError('Layer ' + self.name +
                           ' is not connected, no input to return.')
    return self._get_node_attribute_at_index(0, 'input_tensors',
                                             'input')

  @property
  def output(self):
    """Retrieves the output tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    """
    if len(self.inbound_nodes) == 0:
      raise AttributeError('Layer ' + self.name +
                           ' has no inbound nodes.')
    if len(self.inbound_nodes) > 1:
      raise AttributeError('Layer ' + self.name +
                           ' has multiple inbound nodes, '
                           'hence the notion of "layer output" '
                           'is ill-defined. '
                           'Use `get_output_at(node_index)` instead.')
    return self._get_node_attribute_at_index(0, 'output_tensors',
                                               'output')

  @property
  def input_mask(self):
    """Retrieves the input mask tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    """
    if len(self.inbound_nodes) != 1:
      raise AttributeError('Layer ' + self.name +
                           ' has multiple inbound nodes, ' +
                           'hence the notion of "layer input mask" '
                           'is ill-defined. '
                           'Use `get_input_mask_at(node_index)` instead.')
    return self._get_node_attribute_at_index(0, 'input_masks',
                                             'input mask')

  @property
  def output_mask(self):
    """Retrieves the output mask tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    """
    if len(self.inbound_nodes) != 1:
      raise AttributeError('Layer ' + self.name +
                           ' has multiple inbound nodes, '
                           'hence the notion of "layer output mask" '
                           'is ill-defined. '
                           'Use `get_output_mask_at(node_index)` '
                           'instead.')
    return self._get_node_attribute_at_index(0, 'output_masks',
                                             'output mask')

  @property
  def input_shape(self):
    """Retrieves the input shape tuple(s) of a layer. Only applicable
    if the layer has one inbound node,
    or if all inbound nodes have the same input shape.
    """
    if not self.inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined input shape.')
    all_input_shapes = set([str(node.input_shapes) for node in self.inbound_nodes])
    if len(all_input_shapes) == 1:
      input_shapes = self.inbound_nodes[0].input_shapes
      if len(input_shapes) == 1:
        return input_shapes[0]
      else:
        return input_shapes
    else:
      raise AttributeError('The layer "' + str(self.name) +
                           ' has multiple inbound nodes, '
                           'with different input shapes. Hence '
                           'the notion of "input shape" is '
                           'ill-defined for the layer. '
                           'Use `get_input_shape_at(node_index)` '
                           'instead.')

  @property
  def output_shape(self):
    """Retrieves the output shape tuple(s) of a layer. Only applicable
    if the layer has one inbound node,
    or if all inbound nodes have the same output shape.
    """
    if not self.inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined output shape.')
    all_output_shapes = set([str(node.output_shapes) for node in self.inbound_nodes])
    if len(all_output_shapes) == 1:
      output_shapes = self.inbound_nodes[0].output_shapes
      if len(output_shapes) == 1:
        return output_shapes[0]
      else:
        return output_shapes
    else:
      raise AttributeError('The layer "' + str(self.name) +
                           ' has multiple inbound nodes, '
                           'with different output shapes. Hence '
                           'the notion of "output shape" is '
                           'ill-defined for the layer. '
                           'Use `get_output_shape_at(node_index)` '
                           'instead.')

  def add_loss(self, losses, inputs=None):
    if losses is None:
      return
    # Update self.losses
    losses = to_list(losses)
    if not hasattr(self, 'losses'):
      self.losses = []
    try:
      self.losses += losses
    except AttributeError:
      # In case self.losses isn't settable
      # (i.e. it's a getter method).
      # In that case the `losses` property is
      # auto-computed and shouldn't be set.
      pass
    # Update self._per_input_updates
    if not hasattr(self, '_per_input_losses'):
      self._per_input_losses = {}
    if inputs is not None:
      inputs_hash = object_list_uid(inputs)
    else:
      # Updates indexed by None are unconditional
      # rather than input-dependent
      inputs_hash = None
    if inputs_hash not in self._per_input_losses:
      self._per_input_losses[inputs_hash] = []
    self._per_input_losses[inputs_hash] += losses

  def add_update(self, updates, inputs=None):
    if updates is None:
      return
    # Update self.updates
    updates = to_list(updates)
    if not hasattr(self, 'updates'):
      self.updates = []
    try:
      self.updates += updates
    except AttributeError:
      # In case self.updates isn't settable
      # (i.e. it's a getter method).
      # In that case the `updates` property is
      # auto-computed and shouldn't be set.
      pass
    # Update self._per_input_updates
    if not hasattr(self, '_per_input_updates'):
      self._per_input_updates = {}
    if inputs is not None:
      inputs_hash = object_list_uid(inputs)
    else:
      # Updates indexed by None are unconditional
      # rather than input-dependent
      inputs_hash = None
    if inputs_hash not in self._per_input_updates:
      self._per_input_updates[inputs_hash] = []
    self._per_input_updates[inputs_hash] += updates

  def get_updates_for(self, inputs):
    if not hasattr(self, '_per_input_updates'):
      return []
    if inputs is not None:
      inputs_hash = object_list_uid(inputs)
    else:
      inputs_hash = None
    if inputs_hash in self._per_input_updates:
      return self._per_input_updates[inputs_hash]
    return []

  def get_losses_for(self, inputs):
    if not hasattr(self, '_per_input_losses'):
      return []
    if inputs is not None:
      inputs_hash = object_list_uid(inputs)
    else:
      inputs_hash = None
    if inputs_hash in self._per_input_losses:
      return self._per_input_losses[inputs_hash]
    return []

  @property
  def weights(self):
    return self.trainable_weights + self.non_trainable_weights

  def set_weights(self, weights):
    """Sets the weights of the layer, from Numpy arrays.

    # Arguments
      weights: a list of Numpy arrays. The number
        of arrays and their shape must match
        number of the dimensions of the weights
        of the layer (i.e. it should match the
        output of `get_weights`).
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError('You called `set_weights(weights)` on layer "' +
                       self.name +
                       '" with a  weight list of length ' +
                       str(len(weights)) +
                       ', but the layer was expecting ' +
                       str(len(params)) +
                       ' weights. Provided weights: ' +
                       str(weights)[:50] + '...')
    if not params:
        return
    weight_value_tuples = []
    param_values = K.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError('Layer weight shape ' +
                         str(pv.shape) +
                         ' not compatible with '
                         'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    K.batch_set_value(weight_value_tuples)

  def get_weights(self):
    """Returns the current weights of the layer,
    as a list of numpy arrays.
    """
    params = self.weights
    return K.batch_get_value(params)

class InputLayer(Layer):
    """Layer to be used as an entry point into a graph.
    It can either wrap an existing tensor (pass an `input_tensor` argument)
    or create its a placeholder tensor (pass arguments `input_shape`
    or `batch_input_shape` as well as `input_dtype`).
    # Arguments
        input_shape: Shape tuple, not including the batch axis.
        batch_input_shape: Shape tuple, including the batch axis.
        input_dtype: Datatype of the input.
        input_tensor: Optional tensor to use as layer input
            instead of creating a placeholder.
        name: Name of the layer (string).
    """

    def __init__(self, input_shape=None, batch_input_shape=None,
                 input_dtype=None, input_tensor=None, name=None):
      self.input_spec = None
      self.supports_masking = False
      self.uses_learning_phase = False
      self.trainable = False
      self.built = True
      self._trainable_weights = []
      self._non_trainable_weights = []
      self.inbound_nodes = []
      self.outbound_nodes = []
      self.constraints = {}

      if not name:
        prefix = 'input'
        # TODO(rbharath): Keras uses a global var here to maintain
        # unique counts. This seems dangerous. How does tensorflow handle?
        name = prefix + '_' + str(K.get_uid(prefix))
      self.name = name

      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      if input_tensor is not None:
        # Attempt automatic input shape inference.
        try:
          batch_input_shape = K.int_shape(input_tensor)
        except:
          if not input_shape and not batch_input_shape:
            raise ValueError('InputLayer was provided '
                             'an input_tensor argument, '
                             'but its input shape cannot be '
                             'automatically inferred. '
                             'You should pass an input_shape or '
                             'batch_input_shape argument.')
      if not batch_input_shape:
        if not input_shape:
          raise ValueError('An Input layer should be passed either '
                           'a `batch_input_shape` or an `input_shape`.')
        else:
          batch_input_shape = (None,) + tuple(input_shape)
      else:
        batch_input_shape = tuple(batch_input_shape)

      if not input_dtype:
        if input_tensor is None:
          input_dtype = tf.float32
        else:
          input_dtype = dtype(input_tensor)

      self.batch_input_shape = batch_input_shape
      self.input_dtype = input_dtype

      if input_tensor is None:
        input_tensor = tf.placeholder(dtype=input_dtype,
                                      shape=batch_input_shape,
                                      name=self.name)
      else:
        input_tensor._keras_shape = batch_input_shape
      # Create an input node to add to self.outbound_node
      # and set output_tensors' _keras_history.
      input_tensor._uses_learning_phase = False
      input_tensor._keras_history = (self, 0, 0)
      Node(self,
           inbound_layers=[],
           node_indices=[],
           tensor_indices=[],
           input_tensors=[input_tensor],
           output_tensors=[input_tensor],
           input_masks=[None],
           output_masks=[None],
           input_shapes=[batch_input_shape],
           output_shapes=[batch_input_shape])

def Input(shape=None, batch_shape=None,
          name=None, dtype=tf.float32, tensor=None):
  """`Input()` is used to instantiate a Keras tensor.
  A Keras tensor is a tensor object from the underlying backend
  (TensorFlow), which we augment with certain
  attributes that allow us to build a Keras model
  just by knowing the inputs and outputs of the model.
  For instance, if a, b and c and Keras tensors,
  it becomes possible to do:
  `model = Model(input=[a, b], output=c)`
  The added Keras attributes are:
      ._keras_shape: Integer shape tuple propagated
          via Keras-side shape inference.
      ._keras_history: Last layer applied to the tensor.
          the entire layer graph is retrievable from that layer,
          recursively.
  # Arguments
      shape: A shape tuple (integer), not including the batch size.
          For instance, `shape=(32,)` indicates that the expected input
          will be batches of 32-dimensional vectors.
      batch_shape: A shape tuple (integer), including the batch size.
          For instance, `batch_shape=(10, 32)` indicates that
          the expected input will be batches of 10 32-dimensional vectors.
          `batch_shape=(None, 32)` indicates batches of an arbitrary number
          of 32-dimensional vectors.
      name: An optional name string for the layer.
          Should be unique in a model (do not reuse the same name twice).
          It will be autogenerated if it isn't provided.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
  # Example
      ```python
      # this is a logistic regression in Keras
      a = Input(shape=(32,))
      b = Dense(16, activation='softmax')(a)
      model = Model(input=a, output=b)
      ```
  """
  if not batch_shape and tensor is None:
    assert shape, ('Please provide to Input either a `shape`'
                   ' or a `batch_shape` argument. Note that '
                   '`shape` does not include the batch '
                   'dimension.')
  if shape and not batch_shape:
    batch_shape = (None,) + tuple(shape)
  input_layer = InputLayer(batch_input_shape=batch_shape,
                           name=name, input_dtype=dtype,
                           input_tensor=tensor)
  # Return tensor including _keras_shape and _keras_history.
  # Note that in this case train_output and test_output are the same pointer.
  outputs = input_layer.inbound_nodes[0].output_tensors
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs

class Dense(Layer):
  """Just your regular densely-connected NN layer.

  # Example

  ```python
      # as first layer in a sequential model:
      model = Sequential()
      model.add(Dense(32, input_dim=16))
      # now the model will take as input arrays of shape (*, 16)
      # and output arrays of shape (*, 32)

      # this is equivalent to the above:
      model = Sequential()
      model.add(Dense(32, input_shape=(16,)))

      # after the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(Dense(32))
  ```

  # Arguments
    output_dim: int > 0.
    init: name of initialization function for the weights of the layer
      (see [initializations](../initializations.md)),.
      This parameter is only relevant
      if you don't pass a `weights` argument.
    activation: name of activation function to use
      (see [activations](../activations.md)).
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: a(x) = x).
    weights: list of Numpy arrays to set as initial weights.
      The list should have 2 elements, of shape `(input_dim, output_dim)`
      and (output_dim,) for weights and biases respectively.
    W_regularizer: instance of [WeightRegularizer](../regularizers.md)
      (eg. L1 or L2 regularization), applied to the main weights matrix.
    b_regularizer: instance of [WeightRegularizer](../regularizers.md),
      applied to the bias.
    activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
      applied to the network output.
    W_constraint: instance of the [constraints](../constraints.md) module
      (eg. maxnorm, nonneg), applied to the main weights matrix.
    b_constraint: instance of the [constraints](../constraints.md) module,
      applied to the bias.
    bias: whether to include a bias
      (i.e. make the layer affine rather than linear).
    input_dim: dimensionality of the input (integer). This argument
      (or alternatively, the keyword argument `input_shape`)
      is required when using this layer as the first layer in a model.

  # Input shape
    nD tensor with shape: `(nb_samples, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(nb_samples, input_dim)`.

  # Output shape
    nD tensor with shape: `(nb_samples, ..., output_dim)`.
    For instance, for a 2D input with shape `(nb_samples, input_dim)`,
    the output would have shape `(nb_samples, output_dim)`.
  """

  def __init__(self, output_dim, init='glorot_uniform',
               activation=None, weights=None,
               W_regularizer=None, b_regularizer=None, activity_regularizer=None,
               W_constraint=None, b_constraint=None,
               bias=True, input_dim=None, **kwargs):
    self.init = initializations.get(init)
    self.activation = activations.get(activation)
    self.output_dim = output_dim
    self.input_dim = input_dim

    self.W_regularizer = regularizers.get(W_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.W_constraint = constraints.get(W_constraint)
    self.b_constraint = constraints.get(b_constraint)

    self.bias = bias
    self.initial_weights = weights
    self.input_spec = [InputSpec(ndim='2+')]

    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(Dense, self).__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape) >= 2
    input_dim = input_shape[-1]
    self.input_dim = input_dim
    self.input_spec = [InputSpec(dtype=tf.float32,
                                 ndim='2+')]

    self.W = self.add_weight((input_dim, self.output_dim),
                             initializer=self.init,
                             name='{}_W'.format(self.name),
                             regularizer=self.W_regularizer,
                             constraint=self.W_constraint)
    if self.bias:
      self.b = self.add_weight((self.output_dim,),
                               initializer='zero',
                               name='{}_b'.format(self.name),
                               regularizer=self.b_regularizer,
                               constraint=self.b_constraint)
    else:
      self.b = None

    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights
    self.built = True

  def call(self, x, mask=None):
    output = K.dot(x, self.W)
    if self.bias:
      output += self.b
    return self.activation(output)

  def get_output_shape_for(self, input_shape):
    assert input_shape and len(input_shape) >= 2
    assert input_shape[-1] and input_shape[-1] == self.input_dim
    output_shape = list(input_shape)
    output_shape[-1] = self.output_dim
    return tuple(output_shape)

class Dropout(Layer):
  """Applies Dropout to the input.

  Dropout consists in randomly setting
  a fraction `p` of input units to 0 at each update during training time,
  which helps prevent overfitting.

  # Arguments
      p: float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the
          binary dropout mask that will be multiplied with the input.
          For instance, if your inputs ahve shape
          `(batch_size, timesteps, features)` and
          you want the dropout mask to be the same for all timesteps,
          you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.

  # References
      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
  """

  def __init__(self, p, noise_shape=None, seed=None, **kwargs):
    self.p = p
    self.noise_shape = noise_shape
    self.seed = seed
    if 0. < self.p < 1.:
        self.uses_learning_phase = True
    self.supports_masking = True
    super(Dropout, self).__init__(**kwargs)

  def _get_noise_shape(self, _):
    return self.noise_shape

  def call(self, x, mask=None):
    if 0. < self.p < 1.:
      noise_shape = self._get_noise_shape(x)

      def dropped_inputs():
        retain_prob = 1 - self.p
        return tf.nn.dropout(x * 1., retain_prob, noise_shape, seed=self.seed)
      x = K.in_train_phase(dropped_inputs, lambda: x)
    return x

class BatchNormalization(Layer):
  """Batch normalization layer (Ioffe and Szegedy, 2014).

  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  # Arguments
    epsilon: small float > 0. Fuzz parameter.
    mode: integer, 0, 1 or 2.
      - 0: feature-wise normalization.
          Each feature map in the input will
          be normalized separately. The axis on which
          to normalize is specified by the `axis` argument.
          During training we use per-batch statistics to normalize
          the data, and during testing we use running averages
          computed during the training phase.
      - 1: sample-wise normalization. This mode assumes a 2D input.
      - 2: feature-wise normalization, like mode 0, but
          using per-batch statistics to normalize the data during both
          testing and training.
    axis: integer, axis along which to normalize in mode 0. For instance,
      if your input tensor has shape (samples, channels, rows, cols),
      set axis to 1 to normalize per feature map (channels axis).
    momentum: momentum in the computation of the
      exponential average of the mean and standard deviation
      of the data, for feature-wise normalization.
    weights: Initialization weights.
      List of 2 Numpy arrays, with shapes:
      `[(input_shape,), (input_shape,)]`
      Note that the order of this list is [gamma, beta, mean, std]
    beta_init: name of initialization function for shift parameter
      (see [initializations](../initializations.md)), or alternatively,
      TensorFlow function to use for weights initialization.
      This parameter is only relevant if you don't pass a `weights` argument.
    gamma_init: name of initialization function for scale parameter (see
      [initializations](../initializations.md)), or alternatively,
      TensorFlow function to use for weights initialization.
      This parameter is only relevant if you don't pass a `weights` argument.
    gamma_regularizer: instance of [WeightRegularizer](../regularizers.md)
      (eg. L1 or L2 regularization), applied to the gamma vector.
    beta_regularizer: instance of [WeightRegularizer](../regularizers.md),
      applied to the beta vector.

  # Input shape
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  # Output shape
    Same shape as input.

  # References
    - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  def __init__(self, epsilon=1e-3, mode=0, axis=-1, momentum=0.99,
               weights=None, beta_init='zero', gamma_init='one',
               gamma_regularizer=None, beta_regularizer=None, **kwargs):
    self.supports_masking = True
    self.beta_init = initializations.get(beta_init)
    self.gamma_init = initializations.get(gamma_init)
    self.epsilon = epsilon
    self.mode = mode
    self.axis = axis
    self.momentum = momentum
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.initial_weights = weights
    if self.mode == 0:
      self.uses_learning_phase = True
    super(BatchNormalization, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_spec = [InputSpec(shape=input_shape)]
    shape = (input_shape[self.axis],)

    self.gamma = self.add_weight(shape,
                                 initializer=self.gamma_init,
                                 regularizer=self.gamma_regularizer,
                                 name='{}_gamma'.format(self.name))
    self.beta = self.add_weight(shape,
                                initializer=self.beta_init,
                                regularizer=self.beta_regularizer,
                                name='{}_beta'.format(self.name))
    self.running_mean = self.add_weight(shape, initializer='zero',
                                        name='{}_running_mean'.format(self.name),
                                        trainable=False)
    self.running_std = self.add_weight(shape, initializer='one',
                                       name='{}_running_std'.format(self.name),
                                       trainable=False)

    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights
    self.built = True

  def call(self, x, mask=None):
    if self.mode == 0 or self.mode == 2:
      assert self.built, 'Layer must be built before being called'
      input_shape = K.int_shape(x)

      reduction_axes = list(range(len(input_shape)))
      del reduction_axes[self.axis]
      broadcast_shape = [1] * len(input_shape)
      broadcast_shape[self.axis] = input_shape[self.axis]

      x_normed, mean, std = K.normalize_batch_in_training(
          x, self.gamma, self.beta, reduction_axes,
          epsilon=self.epsilon)

      if self.mode == 0:
        self.add_update([K.moving_average_update(self.running_mean, mean, self.momentum),
                         K.moving_average_update(self.running_std, std, self.momentum)], x)

        if sorted(reduction_axes) == range(get_ndim(x))[:-1]:
          x_normed_running = K.batch_normalization(
              x, self.running_mean, self.running_std,
              self.beta, self.gamma,
              epsilon=self.epsilon)
        else:
          # need broadcasting
          broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
          broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
          broadcast_beta = K.reshape(self.beta, broadcast_shape)
          broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
          x_normed_running = K.batch_normalization(
              x, broadcast_running_mean, broadcast_running_std,
              broadcast_beta, broadcast_gamma,
              epsilon=self.epsilon)

        # pick the normalized form of x corresponding to the training phase
        x_normed = K.in_train_phase(x_normed, x_normed_running)

    elif self.mode == 1:
      # sample-wise normalization
      m = K.mean(x, axis=-1, keepdims=True)
      std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
      x_normed = (x - m) / (std + self.epsilon)
      x_normed = self.gamma * x_normed + self.beta
    return x_normed
