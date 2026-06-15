#!/bin/bash
# Wrapper script - sets LD_PRELOAD before launching Python

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

exec /home/karsa-robert/miniconda3/bin/python3 /home/karsa-robert/hermes/STS/scripts/test_mic_level.py "$@"
