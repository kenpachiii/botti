#!/usr/bin/env python3
"""
__main__.py for Botti
To launch Botti as a module
> python -m botti (with Python >= 3.8)
"""

from botti import main
from asyncio import run

if __name__ == '__main__':
    run(main.main())