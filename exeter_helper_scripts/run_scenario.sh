#!/bin/bash

# Create subfolder name
subfolder_name="$1"

# Create output subfolder
mkdir -p "../output/${subfolder_name}/raw"

# Copy ../toyexample into the subfolder
cp -rp ../toyexample/* "../output/${subfolder_name}/"

# Run the scenario
uv run ../src/CDmetaPop.py "../output/${subfolder_name}" RunVars.csv raw/
