#!/bin/bash

bash check_latex.sh

directory="input_fit"
subdirs=("M1" "M2" "M3" "M4" "M5")
all_subdirs_present=true

for subdir in "${subdirs[@]}"; do
    if [ ! -d "$directory/$subdir" ]; then
        all_subdirs_present=false
        break
    fi
done

cd lsd_out

if [ "$all_subdirs_present" = false ]; then
    python analyse_data.py
    python print_samples.py
    python fit_data.py
else
    python fit_data.py
fi

