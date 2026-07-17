"""
Script to generate a matrix based on patch coordinates from PatchVars.csv.

The matrix has rows and columns corresponding to PatchIDs.
A cell (i,j) is 0 if:
    1. Both patch i and patch j have X and Y coordinates satisfying 4 <= X <= 7 and 4 <= Y <= 7
    2. The patches are direct neighbors along the X or Y axis (Manhattan distance = 1)
Otherwise, the cell is 1.

When writing the CSV, values are written as-is (0 for connections, 1 otherwise).
This connects ALL sites in the 4-7 range on both axes (not randomly selected).
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

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

# Create the matrix (list of lists) with 1 as default and 0 on diagonal
matrix = [[1e9] * n_patches for _ in range(n_patches)]
for i in range(n_patches):
    matrix[i][i] = 0

# First, collect all connections where both patches are in 4<=X<=7 and 4<=Y<=7
# and are direct neighbors along X or Y axis
connections = []
for i in range(n_patches):
    x_i = patches[i]["X"]
    y_i = patches[i]["Y"]

    # Only consider patches in the target range
    if not (4 <= x_i <= 7 and 4 <= y_i <= 7):
        continue

    for j in range(i + 1, n_patches):
        x_j = patches[j]["X"]
        y_j = patches[j]["Y"]

        # Only consider patches in the target range
        if not (4 <= x_j <= 7 and 4 <= y_j <= 7):
            continue

        # Check if patches are direct neighbors along X or Y axis only
        if (abs(x_i - x_j) == 1 and y_i == y_j) or (abs(y_i - y_j) == 1 and x_i == x_j):
            connections.append((i, j))

total_connections = len(connections)
print(f"Total undirected connections in 4<=X<=7, 4<=Y<=7 range: {total_connections}")

# Fill the matrix with all connections (both directions for symmetry)
# 0 indicates a connection
zero_count = 0
for i, j in connections:
    matrix[i][j] = 0
    matrix[j][i] = 0
    zero_count += 2

# Save the matrix
output_dir = "../toyexample/cdmats"
output_path = os.path.join(output_dir, "cdmatrix.csv")
os.makedirs(output_dir, exist_ok=True)

# Convert matrix to DataFrame and save using pandas (no value replacement)
matrix_df = pd.DataFrame(matrix)
matrix_df.to_csv(output_path, index=False, header=False)

print(f"Matrix saved to: {output_path}")
print(f"Matrix shape: {n_patches}x{n_patches}")
print(f"Number of zero entries (connections): {zero_count}")

# Print summary of patches in the target range
range_patches = [p for p in patches if 4 <= p["X"] <= 7 and 4 <= p["Y"] <= 7]
print(f"\nPatches with X in [4,7] and Y in [4,7]: {len(range_patches)}")
print("PatchID  X  Y")
for p in range_patches:
    print(f"{p['PatchID']:8} {p['X']:2} {p['Y']:2}")

# Create mapping from patch index to coordinates using the patches data we already have
patchid_to_coord = {p["PatchID"]: (p["X"], p["Y"]) for p in patches}

# Create figure
plt.figure(figsize=(8, 8))

# Plot all points first
x_coords = [p["X"] for p in patches]
y_coords = [p["Y"] for p in patches]
plt.scatter(x_coords, y_coords)

# Highlight the patches in the target range
range_x = [p["X"] for p in range_patches]
range_y = [p["Y"] for p in range_patches]
plt.scatter(range_x, range_y, c="red", s=100, alpha=0.5)

# Draw lines for all connections (where matrix value is 0)
patch_ids = [p["PatchID"] for p in patches]
for r_idx in range(n_patches):
    for c_idx in range(n_patches):
        if matrix[r_idx][c_idx] == 0:
            x1, y1 = patchid_to_coord[patch_ids[r_idx]]
            x2, y2 = patchid_to_coord[patch_ids[c_idx]]
            plt.plot([x1, x2], [y1, y2], "k-", alpha=0.5, linewidth=2)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Patch Coordinates with Dispersal Links (4<=X<=7, 4<=Y<=7)")

# Save as dispersal.png in the same directory as the matrix
dispersal_plot_path = os.path.join(output_dir, "dispersal.png")
plt.savefig(dispersal_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved dispersal plot: {dispersal_plot_path}")
