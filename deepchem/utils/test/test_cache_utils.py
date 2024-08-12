import pytest


@pytest.mark.torch
def test_Cache():
    from deepchem.utils.cache_utils import Cache

    class A:

        def __init__(self):
            self.cache = Cache.get_dummy()

        def foo(self, x):
            return self.cache.cache("foo", lambda: x * x)

    a = A()
    assert a.foo(2) == 4


@pytest.mark.torch
def test_PrefixedCache():
    from deepchem.utils.cache_utils import Cache, _PrefixedCache
    cache = Cache.get_dummy()
    pcache = _PrefixedCache(cache, "prefix.")
    assert pcache.cache("foo", lambda: 1) == 1
    assert pcache.cache("foo", lambda: 2) == 2


@pytest.mark.torch
def test_DummyCache():
    from deepchem.utils.cache_utils import _DummyCache
    cache = _DummyCache()
    assert cache.cache("foo", lambda: 1) == 1
    assert cache.cache("foo", lambda: 2) == 2
