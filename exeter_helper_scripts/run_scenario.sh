#!/bin/bash

# Create subfolder name
if [ -z "$1" ]; then
    echo "Error: subfolder_name cannot be empty. Please provide a scenario name as the first argument."
    exit 1
fi
subfolder_name="$1"

# Create output subfolder
mkdir -p "../output/${subfolder_name}/raw"
mkdir -p "../output/${subfolder_name}/input"

# Copy ../toyexample into the subfolder
cp -rp ../toyexample/* "../output/${subfolder_name}/input/"

# Run the scenario
uv run ../src/CDmetaPOP.py "../output/${subfolder_name}/input" RunVars.csv ../raw/
