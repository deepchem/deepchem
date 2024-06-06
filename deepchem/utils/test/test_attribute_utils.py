"""
Test Attribute Utils
"""


def test_get_attr():
    from deepchem.utils.attribute_utils import get_attr

    class MyClass:

        def __init__(self):
            self.a = 1
            self.b = 2

    obj = MyClass()
    assert get_attr(obj, "a") == 1


def test_set_attr():
    from deepchem.utils.attribute_utils import set_attr

    class MyClass:

        def __init__(self):
            self.a = 1
            self.b = 2

    obj = MyClass()
    set_attr(obj, "a", 3)
    set_attr(obj, "c", 4)
    assert obj.a == 3
    assert obj.c == 4


def test_del_attr():
    from deepchem.utils.attribute_utils import del_attr

    class MyClass:

        def __init__(self):
            self.a = 1
            self.b = 2

    obj = MyClass()
    del_attr(obj, "a")
    alpha = 0
    try:
        obj.a
    except AttributeError:
        alpha = 1  # alpha changes to 1 if not found
    assert alpha == 1
