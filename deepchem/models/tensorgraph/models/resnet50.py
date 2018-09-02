"""
ResNet-50 implementation
Deep Residual Learning for Image Recognition,
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

https://arxiv.org/abs/1512.03385

"""
import tensorflow as tf
import deepchem as dc
from deepchem.models import Sequential
from deepchem.models.tensorgraph.layers import Conv2D, MaxPool2D, AvgPool2D, Conv2DTranspose, Concat, Feature, Label, BatchNorm
from deepchem.models.tensorgraph.layers import SoftMaxCrossEntropy, ReduceMean, SoftMax, ReLU, Add, Flatten, Dense
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

  def identity_block(self, input, kernel_size, filters):
    filters1, filters2, filters3 = filters

    output = Conv2D(
        num_outputs=filters1,
        kernel_size=1,
        activation='linear',
        padding='same',
        in_layers=[input])
    output = BatchNorm(in_layers=[output])
    output = ReLU(output)

    output = Conv2D(
        num_outputs=filters2,
        kernel_size=kernel_size,
        activation='linear',
        padding='same',
        in_layers=[input])
    output = BatchNorm(in_layers=[output])
    output = ReLU(output)

    output = Conv2D(
        num_outputs=filters3,
        kernel_size=1,
        activation='linear',
        padding='same',
        in_layers=[input])
    output = BatchNorm(in_layers=[output])

    output = Add(in_layers=[output, input])
    output = ReLU(output)

    return output

  def conv_block(self, input, kernel_size, filters, strides=2):
    filters1, filters2, filters3 = filters

    output = Conv2D(
        num_outputs=filters1,
        kernel_size=1,
        stride=strides,
        activation='linear',
        padding='same',
        in_layers=[input])
    output = BatchNorm(in_layers=[output])
    output = ReLU(output)

    output = Conv2D(
        num_outputs=filters2,
        kernel_size=kernel_size,
        activation='linear',
        padding='same',
        in_layers=[output])
    output = BatchNorm(in_layers=[output])
    output = ReLU(output)

    output = Conv2D(
        num_outputs=filters3,
        kernel_size=1,
        activation='linear',
        padding='same',
        in_layers=[output])
    output = BatchNorm(in_layers=[output])

    shortcut = Conv2D(
        num_outputs=filters3,
        kernel_size=1,
        stride=strides,
        activation='linear',
        padding='same',
        in_layers=[input])
    shortcut = BatchNorm(in_layers=[shortcut])
    output = Add(in_layers=[shortcut, output])
    output = ReLU(output)

    return output

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
    self.classes = classes

    input = Feature(shape=(None, self.img_rows, self.img_cols, 3))
    labels = Label(shape=(None, self.classes))

    conv1 = Conv2D(
        num_outputs=64,
        kernel_size=7,
        stride=2,
        activation='linear',
        padding='same',
        in_layers=[input])
    bn1 = BatchNorm(in_layers=[conv1])
    ac1 = ReLU(bn1)
    pool1 = MaxPool2D(ksize=[1, 3, 3, 1], in_layers=[bn1])

    cb1 = self.conv_block(pool1, 3, [64, 64, 256], 1)
    id1 = self.identity_block(cb1, 3, [64, 64, 256])
    id1 = self.identity_block(id1, 3, [64, 64, 256])

    cb2 = self.conv_block(id1, 3, [128, 128, 512])
    id2 = self.identity_block(cb2, 3, [128, 128, 512])
    id2 = self.identity_block(id2, 3, [128, 128, 512])
    id2 = self.identity_block(id2, 3, [128, 128, 512])

    cb3 = self.conv_block(id2, 3, [256, 256, 1024])
    id3 = self.identity_block(cb3, 3, [256, 256, 1024])
    id3 = self.identity_block(id3, 3, [256, 256, 1024])
    id3 = self.identity_block(id3, 3, [256, 256, 1024])
    id3 = self.identity_block(cb3, 3, [256, 256, 1024])
    id3 = self.identity_block(id3, 3, [256, 256, 1024])

    cb4 = self.conv_block(id3, 3, [512, 512, 2048])
    id4 = self.identity_block(cb4, 3, [512, 512, 2048])
    id4 = self.identity_block(id4, 3, [512, 512, 2048])

    pool2 = AvgPool2D(ksize=[1, 7, 7, 1], in_layers=[id4])

    flatten = Flatten(in_layers=[pool2])
    dense = Dense(classes, in_layers=[flatten])

    loss = SoftMaxCrossEntropy(in_layers=[labels, dense])
    loss = ReduceMean(in_layers=[loss])
    self.set_loss(loss)
    self.add_output(dense)
