"""
U-Net implementation
 U-Net: Convolutional Networks for Biomedical Image Segmentation,
 Olaf Ronneberger, Philipp Fischer, Thomas Brox

 https://arxiv.org/pdf/1505.04597.pdf

"""
import tensorflow as tf
import deepchem as dc
from deepchem.models import Sequential
from deepchem.models.tensorgraph.layers import Conv2D, MaxPool2D, Conv2DTranspose, Concat, Feature, Label
from deepchem.models.tensorgraph.layers import SoftMaxCrossEntropy, ReduceMean, SoftMax
from deepchem.models import TensorGraph


class UNet(TensorGraph):
  """
  U-Net architecture implementation.
  Parameters
  ----------
  img_rows : int
   number of rows of the image.
  img_cols: int
   number of columns of the image
  filters: list of input
   List of 5 integers to be given which will be the size of the filters in
   each conv layer.
  """

  def __init__(self,
               img_rows=512,
               img_cols=512,
               filters=[64, 128, 256, 512, 1024],
               **kwargs):
    super(UNet, self).__init__(use_queue=False, **kwargs)
    self.img_cols = img_cols
    self.img_rows = img_rows
    self.filters = filters

    input = Feature(shape=(None, self.img_rows, self.img_cols, 3))
    labels = Label(shape=(None, self.img_rows * self.img_cols))

    conv1 = Conv2D(
        num_outputs=self.filters[0],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[input])
    conv1 = Conv2D(
        num_outputs=self.filters[0],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv1])
    pool1 = MaxPool2D(ksize=[1, 2, 2, 1], in_layers=[conv1])

    conv2 = Conv2D(
        num_outputs=self.filters[1],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[pool1])
    conv2 = Conv2D(
        num_outputs=self.filters[1],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv2])
    pool2 = MaxPool2D(ksize=[1, 2, 2, 1], in_layers=[conv2])

    conv3 = Conv2D(
        num_outputs=self.filters[2],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[pool2])
    conv3 = Conv2D(
        num_outputs=self.filters[2],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv3])
    pool3 = MaxPool2D(ksize=[1, 2, 2, 1], in_layers=[conv3])

    conv4 = Conv2D(
        num_outputs=self.filters[3],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[pool3])
    conv4 = Conv2D(
        num_outputs=self.filters[3],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv4])
    pool4 = MaxPool2D(ksize=[1, 2, 2, 1], in_layers=[conv4])

    conv5 = Conv2D(
        num_outputs=self.filters[4],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[pool4])
    conv5 = Conv2D(
        num_outputs=self.filters[4],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv5])

    up6 = Conv2DTranspose(
        num_outputs=self.filters[3], kernel_size=2, stride=2, in_layers=[conv5])
    concat6 = Concat(in_layers=[conv4, up6], axis=3)
    conv6 = Conv2D(
        num_outputs=self.filters[3],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[concat6])

    conv6 = Conv2D(
        num_outputs=self.filters[3],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv6])

    up7 = Conv2DTranspose(
        num_outputs=self.filters[2], kernel_size=2, stride=2, in_layers=[conv6])
    concat7 = Concat(in_layers=[conv3, up7], axis=3)
    conv7 = Conv2D(
        num_outputs=self.filters[2],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[concat7])
    conv7 = Conv2D(
        num_outputs=self.filters[2],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv7])

    up8 = Conv2DTranspose(
        num_outputs=self.filters[1], kernel_size=2, stride=2, in_layers=[conv7])
    concat8 = Concat(in_layers=[conv2, up8], axis=3)
    conv8 = Conv2D(
        num_outputs=self.filters[1],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[concat8])
    conv8 = Conv2D(
        num_outputs=self.filters[1],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv8])

    up9 = Conv2DTranspose(
        num_outputs=self.filters[0], kernel_size=2, stride=2, in_layers=[conv8])
    concat9 = Concat(in_layers=[conv1, up9], axis=3)
    conv9 = Conv2D(
        num_outputs=self.filters[0],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[concat9])
    conv9 = Conv2D(
        num_outputs=self.filters[0],
        kernel_size=3,
        activation='relu',
        padding='same',
        in_layers=[conv9])

    conv10 = Conv2D(
        num_outputs=1, kernel_size=1, activation='sigmoid', in_layers=[conv9])

    loss = SoftMaxCrossEntropy(in_layers=[labels, conv10])
    loss = ReduceMean(in_layers=[loss])
    self.set_loss(loss)
    self.add_output(conv10)
