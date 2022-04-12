import sys
import pytest
import ccxtpro
from unittest.mock import ANY, MagicMock, PropertyMock, patch
from botti.botti import Botti
from botti.position import Position

from tests.test_cache import patch_cache

def patch_botti(mocker):

    mocker.patch('botti.botti.Botti.loop', MagicMock())

    mocker.patch('botti.botti.Botti.key', PropertyMock())
    mocker.patch('botti.botti.Botti.secret', PropertyMock())
    mocker.patch('botti.botti.Botti.password', PropertyMock())
    mocker.patch('botti.botti.Botti.test', PropertyMock())
    mocker.patch('botti.botti.Botti.symbol', PropertyMock())
    mocker.patch('botti.botti.Botti.fee', PropertyMock())
    mocker.patch('botti.botti.Botti.leverage', PropertyMock())
    mocker.patch('botti.botti.Botti.p_t', PropertyMock())
    mocker.patch('botti.botti.Botti.order_book', PropertyMock())
    mocker.patch('botti.botti.Botti.okx', MagicMock())
    mocker.patch('botti.botti.Botti.cache', MagicMock())

    patch_cache(mocker)
    
    botti = Botti(key='cd145a52-e4be-4c66-abbf-bd9679e8f7c1', secret='5D1C5D3AB5FEB24873A66F798D8F7866', password='LJe4HweCQ52SDTII', test=True)
    return botti

@pytest.mark.asyncio
async def test_add_position(mocker):

    botti = patch_botti(mocker)

@pytest.mark.asyncio
async def test_update_position(mocker):

    botti = patch_botti(mocker)

  

@pytest.mark.asyncio
async def test_trailing_entry(mocker):

    botti = patch_botti(mocker)

    mocker.patch("botti.botti.Botti.okx.id", MagicMock(return_value='okx'))
    mocker.patch("botti.botti.Botti.cache.position.status", 'open')

    assert await botti.trailing_entry() == False
    
    mocker.patch("botti.botti.Botti.okx.id", MagicMock(return_value='okx'))
    mocker.patch("botti.botti.Botti.cache.last.close_avg", 100)
    mocker.patch("botti.botti.Botti.p_t", 102)
    mocker.patch("botti.botti.Botti.cache.position.status", 'closed')

    assert await botti.trailing_entry() == True

    mocker.patch("botti.botti.Botti.okx.id", MagicMock(return_value='okx'))
    mocker.patch("botti.botti.Botti.cache.last.close_avg", 100)
    mocker.patch("botti.botti.Botti.p_t", 97)
    mocker.patch("botti.botti.Botti.cache.position.status", 'closed')

    assert await botti.trailing_entry() == True


@pytest.mark.asyncio
async def test_break_even(mocker):

    botti = patch_botti(mocker)
   

def test_market_depth(mocker):

    botti = patch_botti(mocker)
   

@pytest.mark.asyncio
async def test_take_profits(mocker):
    botti = patch_botti(mocker)

    mocker.patch("botti.botti.Botti.okx.id", MagicMock(return_value='okx'))
    mocker.patch("botti.botti.Botti.cache.position.status", 'open')
    mocker.patch("botti.botti.Botti.cache.position.open_amount", 1)
    mocker.patch("botti.botti.Botti.p_t", 106)
    mocker.patch("botti.botti.Botti.cache.position.open_avg", 100)

    assert await botti.take_profits() == True

@pytest.mark.asyncio
async def test_create_order(mocker):

    botti = patch_botti(mocker)
    

if __name__ == '__main__':
    sys.exit(pytest.main())





