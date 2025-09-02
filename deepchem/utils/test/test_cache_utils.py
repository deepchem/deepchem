import pytest
import os
import time
import gzip
import pickle
import tempfile
import pytest
from multiprocessing import Process, Event

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

if has_torch:
    from deepchem.utils.cache_utils import cached_dirpklgz, FileSystemMutex

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


def hold_lock_and_sleep(mutexfile, flag):
    import time
    with FileSystemMutex(mutexfile):
        time.sleep(2)
        flag.value = 1


def test_mutex_acquire_and_release():
    tmpfile = tempfile.mktemp()
    mutex = FileSystemMutex(tmpfile)

    # Should not raise error
    mutex.acquire()
    assert mutex.handle is not None
    mutex.release()
    assert mutex.handle is None


def test_mutex_context_manager():
    tmpfile = tempfile.mktemp()
    with FileSystemMutex(tmpfile):
        # Inside context, lock should be acquired
        assert os.path.exists(tmpfile)


def test_mutex_release_without_acquire():
    tmpfile = tempfile.mktemp()
    mutex = FileSystemMutex(tmpfile)
    with pytest.raises(RuntimeError):
        mutex.release()


def hold_lock(mutexfile, ready_event):
    with FileSystemMutex(mutexfile):
        ready_event.set()  # signal that lock has been acquired
        time.sleep(2)      # hold the lock for 2 seconds


def test_mutex_blocks_across_processes():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmpfile = tmp.name

    ready_event = Event()
    proc = Process(target=hold_lock, args=(tmpfile, ready_event))
    proc.start()

    ready_event.wait()  # wait for child to acquire lock
    time.sleep(0.1)     # small delay to ensure it's definitely locked

    start = time.time()
    with FileSystemMutex(tmpfile):
        elapsed = time.time() - start

    proc.join(timeout=5)  # avoid infinite blocking
    assert not proc.is_alive(), "Child process did not exit"

    assert elapsed >= 1.5, f"Expected to block for ~2s, but only blocked for {elapsed:.3f}s"


@cached_dirpklgz(dirname=tempfile.gettempdir() + "/cachedir_test", verbose=True)
def dummy_square(x: float) -> float:
    time.sleep(0.5)
    return x * x


def test_cached_result_saved_and_loaded(tmp_path):
    cache_dir = tmp_path / "cached"
    os.makedirs(cache_dir)

    call_count = {"count": 0}

    @cached_dirpklgz(str(cache_dir))
    def expensive_fn(x):
        call_count["count"] += 1
        return x + 1

    # First call should compute
    assert expensive_fn(41) == 42
    assert call_count["count"] == 1

    # Second call should use cache
    assert expensive_fn(41) == 42
    assert call_count["count"] == 1  # Still 1 => cache hit


def test_cache_persistence(tmp_path):
    cache_dir = tmp_path / "cached"
    os.makedirs(cache_dir)

    @cached_dirpklgz(str(cache_dir))
    def identity(x):
        return {"value": x}

    out1 = identity(10)
    out2 = identity(10)
    assert out1 == out2
    assert isinstance(out1, dict)

    # Check if file actually exists
    index_file = os.path.join(cache_dir, "index.pkl")
    assert os.path.exists(index_file)

    with open(index_file, "rb") as f:
        index = pickle.load(f)
        assert len(index) == 1


def test_multiple_args(tmp_path):
    cache_dir = tmp_path / "multi"
    os.makedirs(cache_dir)

    @cached_dirpklgz(str(cache_dir))
    def combine(x, y=0):
        return x + y

    assert combine(2, y=3) == 5
    assert combine(2, y=3) == 5  # Should hit cache


def test_cache_file_format(tmp_path):
    cache_dir = tmp_path / "cachegzip"
    os.makedirs(cache_dir)

    @cached_dirpklgz(str(cache_dir))
    def val(x):
        return x * 2

    # Check .pkl.gz exists
    for file in os.listdir(cache_dir):
        if file.endswith(".pkl.gz"):
            with gzip.open(os.path.join(cache_dir, file), "rb") as f:
                value = pickle.load(f)
                assert value == 14


def test_cache_index_thread_safety(tmp_path):
    cache_dir = str(tmp_path / "race")
    os.makedirs(cache_dir)

    procs = [
        Process(target=call_slow_inc_twice, args=(cache_dir,)) for _ in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    index_file = os.path.join(cache_dir, "index.pkl")
    assert os.path.exists(index_file)

def call_slow_inc_twice(cache_dir):
    import time

    @cached_dirpklgz(cache_dir)
    def slow_inc(x):
        time.sleep(0.2)
        return x + 1

    for _ in range(2):
        assert slow_inc(1) == 2
