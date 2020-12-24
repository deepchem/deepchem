import os
import sys
import itertools
import subprocess

import numpy as np

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

try:
  try:
    # Import for pip package
    from model_optimizer import mo_tf
  except:
    # Import for OpenVINO distribution
    import mo_tf
  from openvino.inference_engine import IECore, StatusCode
  is_available = True
except:
  is_available = False
  pass
"""
Class which wraps Intel OpenVINO toolkit for deep learning inference.
To enable optimization, pass `use_openvino=True` flag when create a model.
Read more at https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html
"""


class OpenVINOModel:

  def __init__(self, model, model_dir, batch_size):
    self.ie = IECore() if self.is_available() else None
    self.exec_net = None
    self.model = model
    self.model_dir = model_dir
    self.batch_size = batch_size
    self.outputs = []

  """
  Prepare model for OpenVINO inference:
    1. Freeze model to .pb file
    2. Run Model Optimizer tool to get OpenVINO Intermediate Representation (IR)
    3. Load model to device
  NOTE: We do not load model in __init__ method because of training.
  """

  def _load_model(self):
    # Freeze Keras model
    func = tf.function(lambda x: self.model(x))
    func = func.get_concrete_function(self.model.inputs)
    frozen_func = convert_variables_to_constants_v2(func)
    graph_def = frozen_func.graph.as_graph_def()

    # Set batch size. Remove training inputs
    for i in reversed(range(len(graph_def.node))):
      node = graph_def.node[i]
      if node.op == 'Placeholder':
        if node.name.startswith('unused_control_flow_input'):
          del graph_def.node[i]
        elif node.attr['shape'].shape.dim[0].size == -1:
          node.attr['shape'].shape.dim[0].size = self.batch_size

    # Save frozen graph
    pb_model_path = os.path.join(self.model_dir, 'model.pb')
    with tf.io.gfile.GFile(pb_model_path, 'wb') as f:
      f.write(graph_def.SerializeToString())

    # Convert to OpenVINO IR
    subprocess.run(
        [
            sys.executable, mo_tf.__file__, '--input_model', pb_model_path,
            '--output_dir', self.model_dir
        ],
        check=True)
    os.remove(pb_model_path)

    # Load network to device
    net = self.ie.read_network(
        os.path.join(self.model_dir, 'model.xml'),
        os.path.join(self.model_dir, 'model.bin'))
    self.exec_net = self.ie.load_network(
        net,
        'CPU',
        config={'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'},
        num_requests=0)

  """
  OpenVINO can process data asynchronously.
  Initialize an iterator for data generator and get outputs by readiness.
  """

  def __call__(self, generator, keras_model):
    if not self.exec_net:
      self._load_model()

    assert (len(self.exec_net.input_info) == 1
           ), 'Not implemented: OpenVINO backend for multiple inputs'
    assert (len(self.exec_net.outputs) == 1
           ), 'Not implemented: OpenVINO backend for multiple outputs'
    inp_name = next(iter(self.exec_net.input_info.keys()))
    out_name = next(iter(self.exec_net.outputs.keys()))

    infer_request_input_id = [-1] * len(self.exec_net.requests)

    # Create a copy of the generator so the origin one can iterate again.
    generator_copy, generator = itertools.tee(generator, 2)

    for inp_id, batch in enumerate(generator_copy):
      inputs, labels, weights = batch
      keras_model._create_inputs(inputs)
      inputs, _, _ = keras_model._prepare_batch((inputs, None, None))
      inputs = inputs[0]

      # Last batch size may be less or equal than overall batch size.
      # Pad extra values by zeros and cut at the end.
      last_batch_size = inputs.shape[0]
      if last_batch_size != self.batch_size:
        assert (last_batch_size < self.batch_size)
        inp = np.zeros(
            [self.batch_size] + list(inputs.shape[1:]), dtype=np.float32)
        inp[:last_batch_size] = inputs
        inputs = inp

      # Get idle infer request
      infer_request_id = self.exec_net.get_idle_request_id()
      if infer_request_id < 0:
        status = self.exec_net.wait(num_requests=1)
        if status != StatusCode.OK:
          raise Exception('Wait for idle request failed!')
        infer_request_id = self.exec_net.get_idle_request_id()
        if infer_request_id < 0:
          raise Exception('Invalid request id!')

      out_id = infer_request_input_id[infer_request_id]
      request = self.exec_net.requests[infer_request_id]

      # Copy output prediction (if already started)
      if out_id != -1:
        self.outputs[out_id] = request.output_blobs[out_name].buffer

      infer_request_input_id[infer_request_id] = inp_id

      self.outputs.append(None)
      request.async_infer({inp_name: inputs})

    # Copy rest of outputs
    status = self.exec_net.wait()
    if status != StatusCode.OK:
      raise Exception('Wait for idle request failed!')
    for infer_request_id, out_id in enumerate(infer_request_input_id):
      if self.outputs[out_id] is None:
        request = self.exec_net.requests[infer_request_id]
        output = request.output_blobs[out_name].buffer
        if out_id == len(self.outputs) - 1:
          self.outputs[out_id] = output[:last_batch_size]
        else:
          self.outputs[out_id] = output

    return self, generator

  def __next__(self):
    return self.outputs.pop(0)

  """
  Returns true if OpenVINO is imported correctly and can be used.
  """

  def is_available(self):
    return is_available
