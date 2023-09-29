"""
Test DFT Miscellaneous Utilities.
"""

from deepchem.utils.dft_utils.misc import set_default_option, memoize_method, get_option, logger


def test_set_default_option():
    """Test set_default_option."""
    defopt = {"a": 1, "b": 2}
    opt = {"b": 3, "c": 4}
    result = set_default_option(defopt, opt)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_memoize_method():
    """Test memoize_method."""

    class MyClass:

        def __init__(self):
            self.count = 0

        @memoize_method
        def fcn(self):
            print("fcn called")
            return self.count

    obj = MyClass()
    assert obj.fcn() == 0
    obj.count = 1
    assert obj.fcn() == 0  # memoized


def test_get_option():
    """Test get_option."""
    options = {"a": 1, "b": 2}
    assert get_option("option", "a", options) == 1
    assert get_option("option", "b", options) == 2


def test_logger():
    """Test logger."""
    # Just test that this doesn't crash
    logger.log("test: ", 0)
