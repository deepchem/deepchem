def test_uniquifier():
    from deepchem.utils import Uniquifier
    a = 1
    b = 2
    c = 3
    d = 1
    u = Uniquifier([a, b, c, a, d])
    assert u.get_unique_objs() == [1, 2, 3]


def test_get_option():
    from deepchem.utils import get_option
    options = {"a": 1, "b": 2}
    assert get_option("name", "a", options) == 1

def test_levenstein_distance():
    from deepchem.utils import levenshtein_distance
    assert levenshtein_distance("kitten", "sitting") == 3