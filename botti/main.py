import logging

from botti.loggers import setup_logging_pre
from botti.botti import Botti

api_keys = {
    'main': {
        'key': '56956a7f-34ed-48f0-b113-aa00f45a525e',
        'secret': 'F4BAE9A3844E26FC46693A33582C9D8A',
        'password': 'ZGN7WIPReYUzwdsb',
        'test': False
    },
    'bottie-api': {
        'key': '34dc5ac4-b78b-4dfa-b0ca-3a69e7cedd07',
        'secret': 'DECA9E6C4CDA7D8C8C97AF5A148FAEEC',
        'password': 'KQMlR1m+g85tJMD2',
        'test': False
    },
    'demo': {
        'key': 'cd145a52-e4be-4c66-abbf-bd9679e8f7c1',
        'secret': '5D1C5D3AB5FEB24873A66F798D8F7866',
        'password': 'LJe4HweCQ52SDTII',
        'test': True
    }
}

logger = logging.getLogger('botti')

def main():

    setup_logging_pre()

    botti = Botti(**api_keys['demo'])
    botti.run()
    

