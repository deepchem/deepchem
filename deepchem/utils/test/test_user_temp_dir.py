import os
import stat
import tempfile

import pytest

from deepchem.utils import data_utils

pytestmark = pytest.mark.skipif(not hasattr(os, 'getuid'),
                                reason='per-user directory needs os.getuid')


@pytest.fixture
def temp_base(tmp_path, monkeypatch):
    """Point the system temporary directory at tmp_path."""
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr(data_utils, '_user_temp_dir_fallback', None)
    monkeypatch.delenv('DEEPCHEM_DATA_DIR', raising=False)
    return tmp_path


def _expected(base):
    return os.path.join(str(base), 'deepchem-%d' % os.getuid())


def _assert_private(path):
    st = os.lstat(path)
    assert stat.S_ISDIR(st.st_mode)
    assert st.st_uid == os.getuid()
    assert not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def test_default_is_private_per_user_dir(temp_base):
    path = data_utils.get_data_dir()
    assert path == _expected(temp_base)
    _assert_private(path)
    # Stable across calls, so cached datasets are found again.
    assert data_utils.get_data_dir() == path


def test_env_var_takes_precedence(temp_base, monkeypatch):
    monkeypatch.setenv('DEEPCHEM_DATA_DIR', str(temp_base / 'custom'))
    assert data_utils.get_data_dir() == str(temp_base / 'custom')


def test_writable_by_others_is_not_used(temp_base):
    expected = _expected(temp_base)
    os.mkdir(expected)
    os.chmod(expected, 0o777)
    path = data_utils.get_data_dir()
    assert path != expected
    _assert_private(path)


def test_symlink_is_not_used(temp_base):
    target = temp_base / 'elsewhere'
    target.mkdir()
    expected = _expected(temp_base)
    os.symlink(str(target), expected)
    path = data_utils.get_data_dir()
    assert path != expected
    assert os.path.realpath(path) != str(target)
    _assert_private(path)


def test_file_is_not_used(temp_base):
    expected = _expected(temp_base)
    with open(expected, 'w') as f:
        f.write('')
    path = data_utils.get_data_dir()
    assert path != expected
    _assert_private(path)
