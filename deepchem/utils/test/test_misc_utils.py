def test_uniquifier():
    from deepchem.utils import Uniquifier
    a = 1
    b = 2
    c = 3
    d = 1
    u = Uniquifier([a, b, c, a, d])
    assert u.get_unique_objs() == [1, 2, 3]


def test_normalize_prefix():
    from deepchem.utils.misc_utils import normalize_prefix
    assert normalize_prefix("prefix") == 'prefix.'
    assert normalize_prefix("prefix.") == 'prefix.'
