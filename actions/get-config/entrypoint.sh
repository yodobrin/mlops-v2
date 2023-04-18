#!/bin/sh -l

FILE_NAME=$INPUT_FILE_NAME

python -c "import yaml; data = yaml.safe_load(open('$FILE_NAME')); print(data)"

echo ::set-output name=config::$(python -c "import yaml; data = yaml.safe_load(open('$FILE_NAME')); print(data)")
