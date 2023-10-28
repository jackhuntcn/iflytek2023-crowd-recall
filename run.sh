#!/bin/bash

python 01_data_prepare.py
python 02_make_features.py
python 03_run_catboost.py

zip -r submit submit
