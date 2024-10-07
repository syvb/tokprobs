#!/bin/false "This script should be sourced in a shell, not executed directly"
set -e

# This should be used for testing in a dedicated cloud instance (e.g. RunPod). It should not be used for local testing.

export HF_HOME=/workspace/.cache/huggingface
pip3 install huggingface-hub[cli]
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B --include '*.safetensors'

git config --global user.name 'syvb'
git config --global user.email 'me@iter.ca'
