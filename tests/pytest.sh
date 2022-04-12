#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=botti --cov-config=.coveragerc tests/