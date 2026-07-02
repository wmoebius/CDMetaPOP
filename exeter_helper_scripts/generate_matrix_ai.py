"""
Script to generate a matrix based on patch coordinates from PatchVars.csv.

The matrix has rows and columns corresponding to PatchIDs.
A cell (i,j) is 1 if:
    1. Both patch i and patch j have X and Y coordinates between 1 and grid_size (inclusive)
    2. The patches are direct neighbors along the X or Y axis (Manhattan distance = 1)
    3. The connection is among a randomly selected fraction f of all possible such connections
Otherwise, the cell is 0.

Parameters:
    grid_size: maximum coordinate value for the grid (default: 10)
    f: fraction of possible connections to include (default: 1.0)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate a dispersal matrix from patch coordinates.")
parser.add_argument("--f", type=float, default=1.0, help="Fraction of possible connections to include (default: 1.0)")
args = parser.parse_args()
f = args.f

# Direct path to PatchVars.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
patchvars_path = os.path.join(script_dir, "../toyexample/patchvars/PatchVars.csv")
full_patchvars_path = os.path.abspath(patchvars_path)

print(f"Reading patch coordinates from: {full_patchvars_path}")

# Read PatchVars.csv using pandas
try:
    df = pd.read_csv(full_patchvars_path)
    # Select and convert columns, sort by PatchID
    patches = (
        df[["PatchID", "X", "Y"]].astype(int).sort_values("PatchID").to_dict("records")
    )
except Exception as e:
    print(f"Error reading PatchVars.csv: {e}", file=sys.stderr)
    sys.exit(1)

n_patches = len(patches)
print(f"Found {n_patches} patches")

# Determine grid size from patch coordinates
grid_size = max(max(p["X"] for p in patches), max(p["Y"] for p in patches))
print(f"Grid size determined from patches: {grid_size}")
print(f"Using connection fraction f={f}")

# Create the matrix (list of lists)
matrix = [[0] * n_patches for _ in range(n_patches)]

# First, collect all possible undirected connections (i < j)
possible_connections = []
for i in range(n_patches):
    x_i = patches[i]["X"]
    y_i = patches[i]["Y"]

    for j in range(i + 1, n_patches):
        x_j = patches[j]["X"]
        y_j = patches[j]["Y"]

        # Check if both patches are within the grid
        if (
            1 <= x_i <= grid_size
            and 1 <= y_i <= grid_size
            and 1 <= x_j <= grid_size
            and 1 <= y_j <= grid_size
        ):
            # Check if patches are direct neighbors along X or Y axis only
            if (abs(x_i - x_j) == 1 and y_i == y_j) or (
                abs(y_i - y_j) == 1 and x_i == x_j
            ):
                possible_connections.append((i, j))

# Calculate how many connections to include
total_possible = len(possible_connections)
n_connections = int(f * total_possible)

# Randomly select connections (shuffle and take first n_connections)
np.random.seed(42)  # For reproducibility
np.random.shuffle(possible_connections)
selected_connections = possible_connections[:n_connections]

print(f"Total possible undirected connections: {total_possible}")
print(f"Including fraction f={f}: {n_connections} connections")

# Fill the matrix with selected connections (both directions for symmetry)
non_zero_count = 0
for i, j in selected_connections:
    matrix[i][j] = 1
    matrix[j][i] = 1
    non_zero_count += 2

# Save the matrix
output_dir = "../toyexample/cdmats"
output_path = os.path.join(output_dir, "cdmatrix.csv")
os.makedirs(output_dir, exist_ok=True)

# Convert matrix to DataFrame and save using pandas
matrix_df = pd.DataFrame(matrix)
matrix_df.to_csv(output_path, index=False, header=False)

print(f"Matrix saved to: {output_path}")
print(f"Matrix shape: {n_patches}x{n_patches}")
print(f"Number of non-zero entries: {non_zero_count}")

# Print summary of patches in the grid
grid_patches = [
    p for p in patches if 1 <= p["X"] <= grid_size and 1 <= p["Y"] <= grid_size
]
print(f"\nPatches with X and Y in [1,{grid_size}]: {len(grid_patches)}")
print("PatchID  X  Y")
for p in grid_patches:
    print(f"{p["PatchID"]:8} {p["X"]:2} {p["Y"]:2}")

# Create mapping from patch index to coordinates using the patches data we already have
patchid_to_coord = {p["PatchID"]: (p["X"], p["Y"]) for p in patches}

# Create figure
plt.figure(figsize=(8, 8))

# Plot all points first
x_coords = [p["X"] for p in patches]
y_coords = [p["Y"] for p in patches]
plt.scatter(x_coords, y_coords)

# Find non-zero entries in the generated matrix and draw lines
# Matrix indices correspond to patch indices (sorted by PatchID)
patch_ids = [p["PatchID"] for p in patches]
for r_idx in range(n_patches):
    for c_idx in range(n_patches):
        if matrix[r_idx][c_idx] != 0:
            x1, y1 = patchid_to_coord[patch_ids[r_idx]]
            x2, y2 = patchid_to_coord[patch_ids[c_idx]]
            plt.plot([x1, x2], [y1, y2], "k-", alpha=0.5, linewidth=2)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Patch Coordinates with Dispersal Links")

# Save as dispersal.png in the same directory as the matrix
dispersal_plot_path = os.path.join(output_dir, "dispersal.png")
plt.savefig(dispersal_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved dispersal plot: {dispersal_plot_path}")
