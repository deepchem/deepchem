"""
ResNet-50 implementation
Deep Residual Learning for Image Recognition,
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

https://arxiv.org/abs/1512.03385

"""
import tensorflow as tf
import deepchem as dc
from deepchem.models import Sequential
from deepchem.models.tensorgraph.layers import Conv2D, MaxPool2D, Conv2DTranspose, Concat, Feature, Label, BatchNorm
from deepchem.models.tensorgraph.layers import SoftMaxCrossEntropy, ReduceMean, SoftMax
from deepchem.models import TensorGraph


class ResNet50(TensorGraph):
  """
        ResNet50 architecture implementation.
        Parameters
        ----------
        img_rows : int
         number of rows of the image.
        img_cols: int
         number of columns of the image
        weights: string
         if "imagenet" - weights are initialized with the pretrained values.
        classes: int
         specifies number of classes
    """

  def __init__(self,
               img_rows=224,
               img_cols=224,
               weights="imagenet",
               classes=1000,
               **kwargs):
    super(ResNet50, self).__init__(use_queue=False, **kwargs)
    self.img_cols = img_cols
    self.img_rows = img_rows
    self.weights = weights

    input = Feature(shape=(None, self.img_rows, self.img_cols, 3))
    labels = Label(shape=(None, self.classes))

    conv1 = Conv2D(
        num_outputs=64,
        kernel_size=7,
        stride=2,
        activation='relu',
        padding='same',
        in_layers=[input])
    bn1 = BatchNorm(in_layers=[conv1])
    pool1 = MaxPool2D(ksize=[1, 2, 2, 1], in_layers=[bn1])
