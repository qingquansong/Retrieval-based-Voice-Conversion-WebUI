#!/bin/bash

EXPECTED_DIR="/workspace/Retrieval-based-Voice-Conversion-WebUI"
CURRENT_DIR="$(pwd)"

if [[ "$CURRENT_DIR" != "$EXPECTED_DIR" ]]; then
  echo "❌ This script must be run from: $EXPECTED_DIR"
  echo "Current directory: $CURRENT_DIR"
  exit 1
fi

echo "✅ Directory check passed. Continuing..."

#switch to qsong/test branch
git checkout qsong/test

#create venv and activate it
python -m venv cc_rvc
source cc_rvc/bin/activate

# setup uvr submodule
git submodule set-url ultimatevocalremovergui \
https://qingquansong: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@github.com/qingquansong/ultimatevocalremovergui.git

git submodule update --init --recursive

#install requirements
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
python -m pip install -r requirements.txt --index-url=https://pypi.org/simple

#copy backward incompatible checkpoint_utils.py
cp checkpoint_utils.py /workspace/Retrieval-based-Voice-Conversion-WebUI/cc_rvc/lib/python3.10/site-packages/fairseq/checkpoint_utils.py