import os
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

try:
  import mo_tf
  from openvino.inference_engine import IECore
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
            mo_tf.__file__, '--input_model', pb_model_path, '--output_dir',
            self.model_dir
        ],
        check=True)
    os.remove(pb_model_path)

    # Load network to device
    net = self.ie.read_network(
        os.path.join(self.model_dir, 'model.xml'),
        os.path.join(self.model_dir, 'model.bin'))
    self.exec_net = self.ie.load_network(net, 'CPU')

  """
  Process input data.
  """

  def __call__(self, inputs):
    if not self.exec_net:
      self._load_model()

    assert (len(self.exec_net.input_info) == 1)
    assert (len(self.exec_net.outputs) == 1)
    inp_name = next(iter(self.exec_net.input_info.keys()))
    out_name = next(iter(self.exec_net.outputs.keys()))

    if inputs.shape[0] != self.batch_size:
      assert (inputs.shape[0] < self.batch_size)
      inp = np.zeros(
          [self.batch_size] + list(inputs.shape[1:]), dtype=np.float32)
      inp[:inputs.shape[0]] = inputs
      output = self.exec_net.infer({inp_name: inp})[out_name]
      return output[:inputs.shape[0]]
    else:
      return self.exec_net.infer({inp_name: inputs})[out_name]

  """
  Returns true if OpenVINO is imported correctly and can be used.
  """

  def is_available(self):
    return is_available
