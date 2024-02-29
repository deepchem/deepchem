def test_uniquifier():
    from deepchem.utils import Uniquifier
    a = 1
    b = 2
    c = 3
    d = 1
    u = Uniquifier([a, b, c, a, d])
    assert u.get_unique_objs() == [1, 2, 3]
