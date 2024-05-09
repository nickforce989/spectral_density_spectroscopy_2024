#!/bin/bash

directory="input_fit"

# Check if 'input_fit/' is empty
if [ -z "$(ls -A $directory)" ]; then
    cd lsd_out
    python analyse_data.py
    python print_samples.py
    python fit_data.py
else
    cd lsd_out
    python fit_data.py
fi
