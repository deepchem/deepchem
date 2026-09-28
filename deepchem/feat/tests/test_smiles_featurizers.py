import os
import subprocess
import sys
import unittest

import numpy as np

from deepchem.feat import create_char_to_idx, SmilesToSeq, SmilesToImage

_DATA = os.path.join(os.path.dirname(__file__), "data", "chembl_25_small.csv")


class TestCreateCharToIdx(unittest.TestCase):
    """Tests for the character vocabulary built for Smiles2Vec models."""

    def test_indices_are_contiguous_and_characters_are_sorted(self):
        """The mapping must be a function of the data alone.

        The two special tokens keep the highest indices, and every real
        character is numbered from 0 in sorted order.
        """
        char_to_idx = create_char_to_idx(_DATA, max_len=35)
        characters = [c for c in char_to_idx if c not in ("<pad>", "<unk>")]

        assert [char_to_idx[c] for c in characters
               ] == list(range(len(characters)))
        assert characters == sorted(characters)
        assert sorted([char_to_idx["<pad>"], char_to_idx["<unk>"]
                      ]) == [len(characters),
                             len(characters) + 1]

    def test_mapping_is_identical_across_interpreter_hash_seeds(self):
        """Same CSV, same mapping, regardless of PYTHONHASHSEED.

        CPython randomises str hashing per process, so building the mapping by
        iterating a set gave every process a different char->index assignment.
        A model trained with one mapping and reloaded in another process was
        then fed different token indices than it was trained on, which produces
        wrong predictions rather than an error.

        This has to run out-of-process: within a single interpreter the hash
        seed is fixed, so an in-process comparison cannot see the difference.
        Two seeds are enough -- before the fix, seeds 0 and 1 produced different
        mappings for this fixture's 21 characters.
        """
        program = (
            "import json, sys\n"
            "from deepchem.feat import create_char_to_idx\n"
            "print(json.dumps(create_char_to_idx(sys.argv[1], max_len=35),"
            " sort_keys=True))\n")
        mappings = set()
        for seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run([sys.executable, "-c", program, _DATA],
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                    env=env)
            mappings.add(result.stdout.strip())

        assert len(mappings) == 1, ("char_to_idx changed with PYTHONHASHSEED: "
                                    "{}".format(sorted(mappings)))


class TestSmilesToSeq(unittest.TestCase):
    """Tests for SmilesToSeq featurizers."""

    def setUp(self):
        """Setup."""
        pad_len = 5
        max_len = 35
        filename = _DATA
        char_to_idx = create_char_to_idx(filename, max_len=max_len)
        self.feat = SmilesToSeq(char_to_idx=char_to_idx,
                                max_len=max_len,
                                pad_len=pad_len)

    def test_smiles_to_seq_featurize(self):
        """Test SmilesToSeq featurization."""
        smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
        expected_seq_len = self.feat.max_len + 2 * self.feat.pad_len

        features = self.feat.featurize(smiles)
        assert features.shape[0] == len(smiles)
        assert features.shape[-1] == expected_seq_len

    def test_reconstruct_from_seq(self):
        """Test SMILES reconstruction from features."""
        smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
        features = self.feat.featurize(smiles)
        # not support array style inputs
        reconstructed_smile = self.feat.smiles_from_seq(features[0])
        assert smiles[0] == reconstructed_smile


class TestSmilesToImage(unittest.TestCase):
    """Tests for SmilesToImage featurizers."""

    def setUp(self):
        """Setup."""
        self.smiles = [
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"
        ]
        self.long_molecule_smiles = [
            "CCCCCCCCCCCCCCCCCCCC(=O)OCCCNC(=O)c1ccccc1SSc1ccccc1C(=O)NCCCOC(=O)CCCCCCCCCCCCCCCCCCC"
        ]

    def test_smiles_to_image(self):
        """Test default SmilesToImage"""
        featurizer = SmilesToImage()
        features = featurizer.featurize(self.smiles)
        assert features.shape == (2, 80, 80, 1)

    def test_smiles_to_image_with_res(self):
        """Test SmilesToImage with res"""
        featurizer = SmilesToImage()
        base_features = featurizer.featurize(self.smiles)
        featurizer = SmilesToImage(res=0.6)
        features = featurizer.featurize(self.smiles)
        assert features.shape == (2, 80, 80, 1)
        assert not np.allclose(base_features, features)

    def test_smiles_to_image_with_image_size(self):
        """Test SmilesToImage with image_size"""
        featurizer = SmilesToImage(img_size=100)
        features = featurizer.featurize(self.smiles)
        assert features.shape == (2, 100, 100, 1)

    def test_smiles_to_image_with_max_len(self):
        """Test SmilesToImage with max_len"""
        smiles_length = [len(s) for s in self.smiles]
        assert smiles_length == [26, 25]
        featurizer = SmilesToImage(max_len=25)
        features = featurizer.featurize(self.smiles)
        assert features[0].shape == (0,)
        assert features[1].shape == (80, 80, 1)

    def test_smiles_to_image_with_img_spec(self):
        """Test SmilesToImage with img_spec"""
        featurizer = SmilesToImage()
        base_features = featurizer.featurize(self.smiles)
        featurizer = SmilesToImage(img_spec='engd')
        features = featurizer.featurize(self.smiles)
        assert features.shape == (2, 80, 80, 4)
        assert not np.allclose(base_features, features)

    def test_smiles_to_image_long_molecule(self):
        """Test SmilesToImage for a molecule which does not fit the image"""
        featurizer = SmilesToImage(img_size=80,
                                   res=0.5,
                                   max_len=250,
                                   img_spec="std")
        features = featurizer.featurize(self.long_molecule_smiles)
        assert features.shape == (1, 0)
