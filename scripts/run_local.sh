#!/bin/bash
# STS run script - Speech-to-Speech (no key in file)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
exec /home/karsa-robert/miniconda3/bin/speech-to-speech \
  --mode local \
  --stt nemotron \
  --parakeet_tdt_device auto \
  --tts omnivoice \
  --llm_backend responses-api \
  --model_name "unsloth/gemma-4-E4B-it-GGUF" \
  --responses_api_base_url "http://127.0.0.1:8888/v1" \
  --responses_api_api_key "$STS_API_KEY" \
  --log_level INFO
