import pytest
from deepchem.models import BasicMolGANModel

def test_molgan_initialization():
    model = BasicMolGANModel()
    assert model.generator is not None
    assert model.discriminator is not None

def test_molgan_generation():
    model = BasicMolGANModel()
    samples = model.predict_gan_generator(batch_size=2)
    assert len(samples) == 2
