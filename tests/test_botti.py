import sys
import pytest
import ccxtpro
import json

from unittest.mock import ANY, MagicMock, PropertyMock, patch
from botti.botti import Botti
from botti.position import Position

from tests.test_cache import patch_cache

def read_order_book():
    return json.loads(open('./tests/order_book.json').read())

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

def test_add_position(mocker):
    return

def test_update_position(mocker):
    return

def test_trailing_entry(mocker):    
    return

def test_break_even(mocker):
    return
   
def test_market_depth(mocker):

    mocker.patch('botti.botti.Botti', { 'order_book': PropertyMock(read_order_book()) })

    botti = Botti(key='cd145a52-e4be-4c66-abbf-bd9679e8f7c1', secret='5D1C5D3AB5FEB24873A66F798D8F7866', password='LJe4HweCQ52SDTII', test=True)

    assert botti.market_depth('bids', 39185.7, 332) == 39200.0
       
def test_take_profits(mocker):
    return

@pytest.mark.asyncio
async def test_create_order(mocker):
    return
    

if __name__ == '__main__':
    sys.exit(pytest.main())





