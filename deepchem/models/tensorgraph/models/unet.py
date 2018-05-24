"""
U-Net implementation
 U-Net: Convolutional Networks for Biomedical Image Segmentation,
 Olaf Ronneberger, Philipp Fischer, Thomas Brox

 https://arxiv.org/pdf/1505.04597.pdf

"""
import tensorflow as tf
import deepchem as dc
from deepchem.models import Sequential
from deepchem.models.tensorgraph.layers import Conv2D, MaxPool2D, Conv2DTranspose, Concat, Feature
from deepchem.models import TensorGraph

class UNet(TensorGraph):

 """
 U-Net architecture implementation.
 Parameters
 ----------
 img_rows : int
	 number of rows of the image.
 img_cols: int
	 number of columns of the image.
 """

	def __init__(self,img_rows=512, img_cols=512, **kwargs):
		super(UNet, self).__init__(use_queue=False, **kwargs)
		self.img_cols = img_cols
		self.img_rows = img_rows

	def unet_model(self):
		model = dc.models.TensorGraph()

		input = Feature(shape=(None, self.img_rows, self.img_cols))

		conv1 = Conv2D(num_outputs=32, kernel_size=3, activation='relu', padding='same', in_layers=[input])
		pool1 = MaxPool2D(ksize=2, in_layers=[conv1])

		conv2 = Conv2D(num_outputs=64, kernel_size=3, activation='relu', padding='same', in_layers=[pool1])
		pool2 = MaxPool2D(ksize=2, in_layers=[conv2])

		conv3 = Conv2D(num_outputs=128, kernel_size=3, activation='relu', padding='same', in_layers=[pool2])
		pool3 = MaxPool2D(ksize=2, in_layers=[conv3])

		conv4 = Conv2D(num_outputs=256, kernel_size=3, activation='relu', padding='same', in_layers=[pool3])
		pool4 = MaxPool2D(ksize=2, in_layers=[conv4])

		conv5 = Conv2D(num_outputs=512, kernel_size=3, activation='relu', padding='same', in_layers=[pool4])

		up6 = Conv2DTranspose(num_outputs=256, kernel_size=2, in_layers=[conv5])
		concat6 = Concat(in_layers=[conv4, up6], axis=1)
		conv6 = Conv2D(num_outputs=256, kernel_size=3, activation='relu', padding='same', in_layers=[concat6])

		up7 = Conv2DTranspose(num_outputs=128, kernel_size=2, in_layers=[conv6])
		concat7 = Concat(in_layers=[conv3, up7], axis=1)
		conv7 = Conv2D(num_outputs=128, kernel_size=3, activation='relu', padding='same', in_layers=[concat7])

		up8 = Conv2DTranspose(num_outputs=64, kernel_size=2, in_layers=[conv7])
		concat8 = Concat(in_layers=[conv2, up8], axis=1)
		conv8 = Conv2D(num_outputs=64, kernel_size=3, activation='relu', padding='same', in_layers=[concat8])

		up9 = Conv2DTranspose(num_outputs=32, kernel_size=2, in_layers=[conv8])
		concat9 = Concat(in_layers=[conv1, up9], axis=1)
		conv9 = Conv2D(num_outputs=32, kernel_size=3, activation='relu', padding='same', in_layers=[concat9])

		conv10 = Conv2D(num_outputs=1, kernel_size=1, activation='sigmoid', in_layers=[conv9])

		model.add_output(conv10)
		return model
