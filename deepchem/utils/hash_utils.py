"""
Various utilities around hash functions.
"""
from typing import Callable, Dict, Optional, Tuple, Any, List
import numpy as np
import hashlib


def hash_ecfp(ecfp: str, size: int = 1024) -> int:
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

    Returns
    -------
    ecfp_hash: int
        An int < size representing given ECFP fragment
    """
    bytes_ecfp = ecfp.encode('utf-8')
    md5 = hashlib.md5()
    md5.update(bytes_ecfp)
    digest = md5.hexdigest()
    ecfp_hash = int(digest, 16) % (size)
    return ecfp_hash


def hash_sybyl(sybyl, sybyl_types):
    return (sybyl_types.index(sybyl))


def hash_ecfp_pair(ecfp_pair: Tuple[str, str], size: int = 1024) -> int:
    """Returns an int < size representing that ECFP pair.

    Input must be a tuple of strings. This utility is primarily used for
    spatial contact featurizers. For example, if a protein and ligand
    have close contact region, the first string could be the protein's
    fragment and the second the ligand's fragment. The pair could be
    hashed together to achieve one hash value for this contact region.

    Parameters
    ----------
    ecfp_pair: Tuple[str, str]
        Pair of ECFP fragment strings
    size: int, optional (default 1024)
        Hash to an int in range [0, size)

    Returns
    -------
    ecfp_hash: int
        An int < size representing given ECFP pair.
    """
    ecfp = "%s,%s" % (ecfp_pair[0], ecfp_pair[1])
    bytes_ecfp = ecfp.encode('utf-8')
    md5 = hashlib.md5()
    md5.update(bytes_ecfp)
    digest = md5.hexdigest()
    ecfp_hash = int(digest, 16) % (size)
    return ecfp_hash


def vectorize(hash_function: Callable[[Any, int], int],
              feature_dict: Optional[Dict[int, str]] = None,
              size: int = 1024,
              feature_list: Optional[List] = None) -> np.ndarray:
    """Helper function to vectorize a spatial description from a hash.

    Hash functions are used to perform spatial featurizations in
    DeepChem. However, it's necessary to convert backwards from
    the hash function to feature vectors. This function aids in
    this conversion procedure. It creates a vector of zeros of length
    `size`. It then loops through `feature_dict`, uses `hash_function`
    to hash the stored value to an integer in range [0, size) and bumps
    that index.

    Parameters
    ----------
    hash_function: Function, Callable[[str, int], int]
        Should accept two arguments, `feature`, and `size` and
        return a hashed integer. Here `feature` is the item to
        hash, and `size` is an int. For example, if `size=1024`,
        then hashed values must fall in range `[0, 1024)`.
    feature_dict: Dict, optional (default None)
        Maps unique keys to features computed.
    size: int (default 1024)
        Length of generated bit vector
    feature_list: List, optional (default None)
        List of features.

    Returns
    -------
    feature_vector: np.ndarray
        A numpy array of shape `(size,)`
    """
    feature_vector = np.zeros(size)
    if feature_dict is not None:
        on_channels = [
            hash_function(feature, size)
            for key, feature in feature_dict.items()
        ]
        feature_vector[on_channels] += 1
    elif feature_list is not None:
        feature_vector[0] += len(feature_list)

    return feature_vector
