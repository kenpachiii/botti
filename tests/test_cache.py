import sys
import pytest

from botti.cache import Cache
from botti.position import Position

from unittest.mock import MagicMock, Mock, PropertyMock

def patch_cache(mocker):

    mocker.patch('botti.botti.Cache.init', MagicMock())
    mocker.patch('botti.botti.Cache.insert', MagicMock())
    mocker.patch('botti.botti.Cache.update', MagicMock())
    mocker.patch('botti.botti.Cache.position', PropertyMock(Position({})))
    mocker.patch('botti.botti.Cache.last', PropertyMock(Position({})))

    cache = Cache()
    return cache

def test_insert(mocker):

    cache = patch_cache(mocker)

    return

def test_update(mocker):

    cache = patch_cache(mocker)
    return

def test_position(mocker):

    cache = patch_cache(mocker)
    return

def test_last(mocker):

    cache = patch_cache(mocker)
    return

if __name__ == '__main__':
    sys.exit(pytest.main())