"""
TODO(LESWING) Remove h5py dependency
TODO(LESWING) Remove keras dependency and replace with functional keras API
"""

from deepchem.models import Model
from deepchem.models.autoencoder_models.model import MoleculeVAE
from deepchem.trans.transformers import zinc_charset
import os
from subprocess import call


class TensorflowMoleculeEncoder(Model):
  def __init__(self,
               model_dir=None,
               weights_file="model.h5",
               verbose=True,
               charset=zinc_charset,
               latent_rep_size=292):
    """
        TODO(LESWING) replace charset with num_atom_types
        of other charsets
        :param model_dir:
        :param verbose:
        :param charset: list of chars
        :param latent_rep_size:
        """
    super(TensorflowMoleculeEncoder, self).__init__(
      model_dir=model_dir, verbose=verbose)
    weights_file = os.path.join(model_dir, weights_file)
    if os.path.isfile(weights_file):
      m = MoleculeVAE()
      m.load(charset, weights_file, latent_rep_size=latent_rep_size)
      self.model = m
    else:
      # TODO (LESWING) Lazy Load
      raise ValueError("Model file %s doesn't exist" % weights_file)


  @staticmethod
  def zinc_encoder():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    weights_file = os.path.join(current_dir, "model.h5")

    if not os.path.exists(weights_file):
      wget_command = "wget -c http://karlleswing.com/misc/keras-molecule/model.h5"
      call(wget_command.split())
      mv_cmd = "mv model.h5 %s" % current_dir
      call(mv_cmd.split())
    return TensorflowMoleculeEncoder(model_dir=current_dir)

  def fit(self, dataset, nb_epoch=10, batch_size=50, **kwargs):
    """
        TODO(LESWING) Test
        """
    self.model.autoencoder.fit(
      dataset.X,
      dataset.X,
      shuffle=True,
      nb_epoch=10,
      batch_size=50,
      callbacks=[],
      validation_data=(dataset.y, dataset.y))

  def predict_on_batch(self, X):
    """
        TODO(LESWING) Test
        """
    x_latent = self.model.encoder.predict(X)
    return x_latent



class TensorflowMoleculeDecoder(Model):
  def __init__(self,
               model_dir=None,
               weights_file="model.h5",
               verbose=True,
               charset=zinc_charset,
               latent_rep_size=292):
    """
        TODO(LESWING) Convert Dataset.[X,y,w] from the h5.py format.
        TODO(LESWING) default to a charset constructed via Zinc -- it should be a superset
        of other charsets
        :param model_dir:
        :param verbose:
        :param charset: list of chars
        :param latent_rep_size:
        """
    super(TensorflowMoleculeDecoder, self).__init__(
      model_dir=model_dir, verbose=verbose)
    weights_file = os.path.join(model_dir, weights_file)
    if os.path.isfile(weights_file):
      m = MoleculeVAE()
      m.load(charset, weights_file, latent_rep_size=latent_rep_size)
      self.model = m
    else:
      # TODO (LESWING) Lazy Load
      raise ValueError("Model file %s doesn't exist" % weights_file)

  def fit(self, dataset, nb_epoch=10, batch_size=50, **kwargs):
    """
    TODO(LESWING) Test
    """
    raise ValueError("Only can read in Cached Models")

  @staticmethod
  def zinc_decoder():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    weights_file = os.path.join(current_dir, "model.h5")

    if not os.path.exists(weights_file):
      wget_command = "wget http://karlleswing.com/misc/keras-molecule/model.h5"
      call(wget_command.split())
      mv_cmd = "mv model.h5 %s" % current_dir
      call(mv_cmd.split())
    return TensorflowMoleculeDecoder(model_dir=current_dir)

  def predict_on_batch(self, X):
    """
        TODO(LESWING) Test
        """
    x_latent = self.model.decoder.predict(X)
    return x_latent
