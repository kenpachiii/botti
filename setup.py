from setuptools import setup

setup(
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-mock',
        'pytest-asyncio',
        'pytest-random-order'
    ],
    install_requires=[
        # from requirements.txt
        'ccxtpro',
        'numpy',
        'asyncio',
        'botto3'
    ],
)