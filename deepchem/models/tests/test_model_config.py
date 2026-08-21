import os
import tempfile
import json
import deepchem as dc


class DummyModel(dc.models.Model):

    def __init__(self, a=1, b="test", c=[1, 2, 3], **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c


def test_model_config_save_load():
    # Instantiate a generic model with custom parameters
    model = DummyModel(a=5, b="hello", c=[4, 5])

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")

        # Save model configuration
        model.save_config(config_path)

        # Ensure file exists
        assert os.path.exists(config_path)

        # Manually load the JSON config to check the contents
        with open(config_path, 'r') as f:
            config = json.load(f)
            assert config['a'] == 5
            assert config['b'] == "hello"
            assert config['c'] == [4, 5]

        # Load model from configuration using the class method
        new_model = DummyModel.load_from_config(config_path)

        # Validate that the recovered model has the same attributes
        assert new_model.a == model.a
        assert new_model.b == model.b
        assert new_model.c == model.c
