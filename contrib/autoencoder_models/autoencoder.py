"""
TODO(LESWING) Remove h5py dependency
TODO(LESWING) Remove keras dependency and replace with functional keras API
"""
import warnings
from deepchem.models import Model
from deepchem.models.autoencoder_models.model import MoleculeVAE
from deepchem.feat.one_hot import zinc_charset
from deepchem.utils import download_url
import os
from subprocess import call


class TensorflowMoleculeEncoder(Model):
  """
  Transform molecules from one hot encoding into a latent vector
  representation.
  https://arxiv.org/abs/1610.02415
  """

  def __init__(self,
               model_dir=None,
               weights_file="model.h5",
               verbose=True,
               charset_length=len(zinc_charset),
               latent_rep_size=292):
    """

    Parameters
    ----------
    model_dir: str
      Folder to store cached weights
    weights_file: str
      File to store cached weights in model_dir
    verbose: bool
      True for more logging
    charset_length: int
      Length of one_hot_encoded vectors
    latent_rep_size: int
      How large a 1D Vector for latent representation
    """
    warnings.warn("TensorflowMoleculeEncoder Deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    super(TensorflowMoleculeEncoder, self).__init__(
        model_dir=model_dir, verbose=verbose)
    weights_file = os.path.join(model_dir, weights_file)
    if os.path.isfile(weights_file):
      m = MoleculeVAE()
      m.load(charset_length, weights_file, latent_rep_size=latent_rep_size)
      self.model = m
    else:
      # TODO (LESWING) Lazy Load
      raise ValueError("Model file %s doesn't exist" % weights_file)

  @staticmethod
  def zinc_encoder():
    """
    Returns
    -------
    obj
      An Encoder with weights that were trained on the zinc dataset
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    weights_filename = "zinc_model.h5"
    weights_file = os.path.join(current_dir, weights_filename)

    if not os.path.exists(weights_file):
      download_url("http://karlleswing.com/misc/keras-molecule/model.h5",
                   current_dir)
      mv_cmd = "mv model.h5 %s" % weights_file
      call(mv_cmd.split())
    return TensorflowMoleculeEncoder(
        model_dir=current_dir, weights_file=weights_filename)

  def fit(self, dataset, nb_epoch=10, batch_size=50, **kwargs):
    raise ValueError("Only can read in pre-trained models")

  def predict_on_batch(self, X):
    x_latent = self.model.encoder.predict(X)
    return x_latent


class TensorflowMoleculeDecoder(Model):
  """
  Transform molecules from a latent space feature vector into
  a one hot encoding.
  https://arxiv.org/abs/1610.02415
  """

  def __init__(self,
               model_dir=None,
               weights_file="model.h5",
               verbose=True,
               charset_length=len(zinc_charset),
               latent_rep_size=292):
    """

    Parameters
    ----------
    model_dir: str
      Folder to store cached weights
    weights_file: str
      File to store cached weights in model_dir
    verbose: bool
      True for more logging
    charset_length: int
      Length of one_hot_encoded vectors
    latent_rep_size: int
      How large a 1D Vector for latent representation
    """
    warnings.warn("TensorflowMoleculeDecoder Deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    super(TensorflowMoleculeDecoder, self).__init__(
        model_dir=model_dir, verbose=verbose)
    weights_file = os.path.join(model_dir, weights_file)
    if os.path.isfile(weights_file):
      m = MoleculeVAE()
      m.load(charset_length, weights_file, latent_rep_size=latent_rep_size)
      self.model = m
    else:
      # TODO (LESWING) Lazy Load
      raise ValueError("Model file %s doesn't exist" % weights_file)

  def fit(self, dataset, nb_epoch=10, batch_size=50, **kwargs):
    raise ValueError("Only can read in pre-trained models")

  @staticmethod
  def zinc_decoder():
    """
    Returns
    -------
    obj
      A Decoder with weights that were trained on the zinc dataset
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    weights_filename = "zinc_model.h5"
    weights_file = os.path.join(current_dir, weights_filename)

    if not os.path.exists(weights_file):
      download_url("http://karlleswing.com/misc/keras-molecule/model.h5",
                   current_dir)
      mv_cmd = "mv model.h5 %s" % weights_file
      call(mv_cmd.split())
    return TensorflowMoleculeDecoder(
        model_dir=current_dir, weights_file=weights_filename)

  def predict_on_batch(self, X):
    x_latent = self.model.decoder.predict(X)
    return x_latent
