#!/bin/sh
export TORCH_ENV_PATH=`python3 -c "import os ;import sys; import torch; sys.stdout.write(os.path.dirname(os.path.abspath(torch.__file__))); sys.exit(0)"`
echo $TORCH_ENV_PATH
pip install .