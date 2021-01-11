import os
import io
import sys
import itertools
import subprocess

from typing import Any, Iterable, List, Tuple

import numpy as np

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

try:
  import mo_tf
  from openvino.inference_engine import IECore, StatusCode, ExecutableNetwork
  is_available = True
except:
  is_available = False


class OpenVINOModel:
  """
  Class which wraps Intel OpenVINO toolkit for deep learning inference.
  To enable optimization, pass `use_openvino=True` flag when create a model.
  Read more about OpenVINO at [1].

  This class can be used to optimize inference for Keras or PyTorch models.

  Examples
  --------
  PyTorch model with OpenVINO backend:

  >> pytorch_model = torch.nn.Sequential(
  >>    torch.nn.Linear(100, 1000),
  >>    torch.nn.Tanh(),
  >>    torch.nn.Linear(1000, 1))
  >> model = TorchModel(pytorch_model, loss=dc.models.losses.L2Loss(), use_openvino=True)
  >> model.predict(dataset)

  Keras based model with OpenVINO backend (a model from tox21_tf_progressive):

  >> model = dc.models.ProgressiveMultitaskClassifier(
  >>           n_tasks=12,
  >>           n_features=1024,
  >>           layer_sizes=[1000],
  >>           dropouts=[.25],
  >>           learning_rate=0.001,
  >>           batch_size=50,
  >>           use_openvino=True)

  References
  ----------
  .. [1] https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html

  Notes
  -----
  This class is internal.
  """

  def __init__(self,
               model_dir: str,
               batch_size: int,
               keras_model=None,
               torch_model=None):
    self._ie = IECore() if self.is_available() else None
    self._exec_net: ExecutableNetwork = None
    self._keras_model = keras_model
    self._torch_model = torch_model
    self._model_dir = model_dir
    self._batch_size = batch_size
    self._outputs: List = []

  def _read_torch_model(self, generator: Iterable[Tuple[Any, Any, Any]]):
    """
    Prepare PyTorch model for OpenVINO inference:
      1. Convert into ONNX format
      2. Read network from memory
    NOTE: We do not load model in __init__ method because of training.
    """
    import torch

    # We need to serialize ONNX model with real input shape.
    # So we create a copy of the generator to take first input.
    generator_copy, generator = itertools.tee(generator, 2)
    inputs, _, _ = next(generator_copy)
    assert (len(inputs) == 1), 'Not implemented'
    inp_shape = list(inputs[0].shape)
    inp_shape[0] = self._batch_size

    buf = io.BytesIO()
    inp = torch.randn(inp_shape)
    torch.onnx.export(self._torch_model.model, inp, buf, opset_version=11)

    # Import network from memory buffer
    return self._ie.read_network(buf.getvalue(), b'', init_from_buffer=True), \
           generator

  def _read_tf_model(self):
    """
    Prepare TensorFlow/Keras model for OpenVINO inference:
      1. Freeze model to .pb file
      2. Run Model Optimizer tool to get OpenVINO Intermediate Representation (IR)
    NOTE: We do not load model in __init__ method because of training.
    """
    # Freeze Keras model
    model = self._keras_model.model
    func = tf.function(lambda x: model(x))
    func = func.get_concrete_function(model.inputs)
    frozen_func = convert_variables_to_constants_v2(func)
    graph_def = frozen_func.graph.as_graph_def()

    # Set batch size. Remove training inputs
    for i in reversed(range(len(graph_def.node))):
      node = graph_def.node[i]
      if node.op == 'Placeholder':
        if node.name.startswith('unused_control_flow_input'):
          del graph_def.node[i]
        elif node.attr['shape'].shape.dim[0].size == -1:
          node.attr['shape'].shape.dim[0].size = self._batch_size

    # Save frozen graph
    pb_model_path = os.path.join(self._model_dir, 'model.pb')
    with tf.io.gfile.GFile(pb_model_path, 'wb') as f:
      f.write(graph_def.SerializeToString())

    # Convert to OpenVINO IR
    subprocess.run(
        [
            sys.executable, mo_tf.__file__, '--input_model', pb_model_path,
            '--output_dir', self._model_dir
        ],
        check=True)
    os.remove(pb_model_path)

    return self._ie.read_network(
        os.path.join(self._model_dir, 'model.xml'),
        os.path.join(self._model_dir, 'model.bin'))

  def _load_model(self, generator: Iterable[Tuple[Any, Any, Any]]):
    """
    Prepare native model for OpenVINO inference.

    Parameters
    ----------
    generator: Iterable[Tuple[Any, Any, Any]]
      Input data generator is used to get input shapes

    Returns
    -------
    Iterable[Tuple[Any, Any, Any]]
      A copy of the generator whih starts from the beginning.
    """
    assert (self.is_available())
    if self._keras_model is not None:
      net = self._read_tf_model()
    elif self._torch_model is not None:
      net, generator = self._read_torch_model(generator)

    # Load network to the device
    self._exec_net = self._ie.load_network(
        net,
        'CPU',
        config={'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'},
        num_requests=0)
    return generator

  def __call__(self, generator: Iterable[Tuple[Any, Any, Any]]):
    """
    OpenVINO can process data asynchronously.
    Use this method to start data processing.

    Parameters
    ----------
    generator: Iterable[Tuple[Any, Any, Any]]
      Input data generator

    Returns
    -------
    Iterable[Tuple[Any, Any, Any]]
      A copy of the generator whih starts from the beginning.
    """
    if not self._exec_net:
      generator = self._load_model(generator)

    assert (len(self._exec_net.input_info) == 1), 'Not implemented'
    assert (len(self._exec_net.outputs) == 1), 'Not implemented'
    inp_name = next(iter(self._exec_net.input_info.keys()))
    out_name = next(iter(self._exec_net.outputs.keys()))

    infer_request_input_id = [-1] * len(self._exec_net.requests)

    # Create a copy of the generator so the origin one can iterate again.
    generator_copy, generator = itertools.tee(generator, 2)

    for inp_id, batch in enumerate(generator_copy):
      inputs, labels, weights = batch
      if self._keras_model is not None:
        self._keras_model._create_inputs(inputs)
        inputs, _, _ = self._keras_model._prepare_batch((inputs, None, None))
      elif self._torch_model is not None:
        inputs, _, _ = self._torch_model._prepare_batch((inputs, None, None))
      inputs = inputs[0]

      # Last batch size may be less or equal than overall batch size.
      # Pad extra values by zeros and cut at the end.
      last_batch_size = inputs.shape[0]
      if last_batch_size != self._batch_size:
        assert (last_batch_size < self._batch_size)
        inp = np.zeros(
            [self._batch_size] + list(inputs.shape[1:]), dtype=np.float32)
        inp[:last_batch_size] = inputs
        inputs = inp

      # Get idle infer request
      infer_request_id = self._exec_net.get_idle_request_id()
      if infer_request_id < 0:
        status = self._exec_net.wait(num_requests=1)
        if status != StatusCode.OK:
          raise Exception('Wait for idle request failed!')
        infer_request_id = self._exec_net.get_idle_request_id()
        if infer_request_id < 0:
          raise Exception('Invalid request id!')

      out_id = infer_request_input_id[infer_request_id]
      request = self._exec_net.requests[infer_request_id]

      # Copy output prediction (if already started)
      if out_id != -1:
        self._outputs[out_id] = request.output_blobs[out_name].buffer

      infer_request_input_id[infer_request_id] = inp_id

      self._outputs.append(None)
      request.async_infer({inp_name: inputs})

    # Copy rest of outputs
    status = self._exec_net.wait()
    if status != StatusCode.OK:
      raise Exception('Wait for idle request failed!')
    for infer_request_id, out_id in enumerate(infer_request_input_id):
      if self._outputs[out_id] is None:
        request = self._exec_net.requests[infer_request_id]
        output = request.output_blobs[out_name].buffer
        if out_id == len(self._outputs) - 1:
          self._outputs[out_id] = output[:last_batch_size]
        else:
          self._outputs[out_id] = output

    return self, generator

  def __next__(self):
    return self._outputs.pop(0)

  def is_available(self):
    """
    Returns True if OpenVINO is imported correctly and can be used.
    """
    return is_available
