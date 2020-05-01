"""
Various utilities around hash functions.
"""
import logging
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


def hash_ecfp(ecfp, size):
  """
  Returns an int < size representing given ECFP fragment.

  Input must be a string. This utility function is used for various
  ECFP based fingerprints.

  Parameters
  ----------
  ecfp: str
    String to hash. Usually an ECFP fragment.
  size: int, optional (default 1024)
    Hash to an int in range [0, size) 
  """
  ecfp = ecfp.encode('utf-8')
  md5 = hashlib.md5()
  md5.update(ecfp)
  digest = md5.hexdigest()
  ecfp_hash = int(digest, 16) % (size)
  return (ecfp_hash)


def hash_ecfp_pair(ecfp_pair, size):
  """Returns an int < size representing that ECFP pair.

  Input must be a tuple of strings. This utility is primarily used for
  spatial contact featurizers. For example, if a protein and ligand
  have close contact region, the first string could be the protein's
  fragment and the second the ligand's fragment. The pair could be
  hashed together to achieve one hash value for this contact region.

  Parameters
  ----------
  ecfp_pair: tuple
    Pair of ECFP fragment strings
  size: int, optional (default 1024)
    Hash to an int in range [0, size) 
  """
  ecfp = "%s,%s" % (ecfp_pair[0], ecfp_pair[1])
  ecfp = ecfp.encode('utf-8')
  md5 = hashlib.md5()
  md5.update(ecfp)
  digest = md5.hexdigest()
  ecfp_hash = int(digest, 16) % (size)
  return (ecfp_hash)


def vectorize(hash_function, feature_dict=None, size=1024):
  """Helper function to vectorize a spatial description from a hash.

  Hash functions are used to perform spatial featurizations in
  DeepChem. However, it's necessary to convert backwards from
  the hash function to feature vectors. This function aids in
  this conversion procedure. It creates a vector of zeros of length
  `seize`. It then loops through `feature_dict`, uses `hash_function`
  to hash the stored value to an integer in range [0, size) and bumps
  that index.

  Parameters
  ----------
  hash_function: function
    Should accept two arguments, `feature`, and `size` and
    return a hashed integer. Here `feature` is the item to
    hash, and `size` is an int. For example, if `size=1024`,
    then hashed values must fall in range `[0, 1024)`.
  feature_dict: dict
    Maps unique keys to features computed. 
  size: int, optional (default 1024)
    Length of generated bit vector
  """
  feature_vector = np.zeros(size)
  if feature_dict is not None:
    on_channels = [
        hash_function(feature, size) for key, feature in feature_dict.items()
    ]
    feature_vector[on_channels] += 1

  return feature_vector
