import warnings
from keras import backend as K
from keras import objectives
from keras.layers import Input, Lambda
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Flatten, RepeatVector
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Model


class MoleculeVAE():

  autoencoder = None

  def __init__(self):
    warnings.warn("Deprecated. Will be removed in DeepChem 1.4.",
                  DeprecationWarning)

  def create(self,
             charset_length,
             max_length=120,
             latent_rep_size=292,
             weights_file=None):
    x = Input(shape=(max_length, charset_length))
    _, z = self._buildEncoder(x, latent_rep_size, max_length)
    self.encoder = Model(x, z)

    encoded_input = Input(shape=(latent_rep_size,))
    self.decoder = Model(encoded_input,
                         self._buildDecoder(encoded_input, latent_rep_size,
                                            max_length, charset_length))

    x1 = Input(shape=(max_length, charset_length))
    vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
    self.autoencoder = Model(x1,
                             self._buildDecoder(z1, latent_rep_size, max_length,
                                                charset_length))

    if weights_file:
      self.autoencoder.load_weights(weights_file)
      self.encoder.load_weights(weights_file, by_name=True)
      self.decoder.load_weights(weights_file, by_name=True)

    self.autoencoder.compile(
        optimizer='Adam', loss=vae_loss, metrics=['accuracy'])

  def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std=0.01):
    h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
    h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
    h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
    h = Flatten(name='flatten_1')(h)
    h = Dense(435, activation='relu', name='dense_1')(h)

    def sampling(args):
      z_mean_, z_log_var_ = args
      batch_size = K.shape(z_mean_)[0]
      epsilon = K.random_normal(
          shape=(batch_size, latent_rep_size), mean=0., std=epsilon_std)
      return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

    z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
    z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

    def vae_loss(x, x_decoded_mean):
      x = K.flatten(x)
      x_decoded_mean = K.flatten(x_decoded_mean)
      xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
      kl_loss = -0.5 * K.mean(
          1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
      return xent_loss + kl_loss

    return (vae_loss, Lambda(
        sampling, output_shape=(latent_rep_size,),
        name='lambda')([z_mean, z_log_var]))

  def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
    h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
    h = RepeatVector(max_length, name='repeat_vector')(h)
    h = GRU(501, return_sequences=True, name='gru_1')(h)
    h = GRU(501, return_sequences=True, name='gru_2')(h)
    h = GRU(501, return_sequences=True, name='gru_3')(h)
    return TimeDistributed(
        Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

  def save(self, filename):
    self.autoencoder.save_weights(filename)

  def load(self, charset_length, weights_file, latent_rep_size=292):
    self.create(
        charset_length,
        weights_file=weights_file,
        latent_rep_size=latent_rep_size)
