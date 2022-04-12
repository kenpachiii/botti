import unittest
import pytest
import sys
import numpy as np

from botti.retrier import retrier

class TestRetrier:

    def test_retrier(self):

        @retrier
        def mock():
            try:
                return
            except Exception as e:
                return
        return
       

if __name__ == '__main__':
    sys.exit(pytest.main())