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

import csv
import numpy as np
import os
import sys

def read_csv(filepath):
    """Read a CSV file and return a list of dicts."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def main(grid_size=10, f=1.0):
    # Direct path to PatchVars.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    patchvars_path = os.path.join(script_dir, '../toyexample/patchvars/PatchVars.csv')
    full_patchvars_path = os.path.abspath(patchvars_path)
    
    print(f"Reading patch coordinates from: {full_patchvars_path}")
    
    # Read PatchVars.csv
    try:
        patchvars_data = read_csv(full_patchvars_path)
    except Exception as e:
        print(f"Error reading PatchVars.csv: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract PatchID, X, Y and sort by PatchID
    patches = []
    for row in patchvars_data:
        patches.append({
            'PatchID': int(row['PatchID']),
            'X': int(row['X']),
            'Y': int(row['Y'])
        })
    
    patches.sort(key=lambda p: p['PatchID'])
    n_patches = len(patches)
    print(f"Found {n_patches} patches")
    
    # Create the matrix (list of lists)
    matrix = [[0] * n_patches for _ in range(n_patches)]
    
    # First, collect all possible undirected connections (i < j)
    possible_connections = []
    for i in range(n_patches):
        x_i = patches[i]['X']
        y_i = patches[i]['Y']
        
        for j in range(i + 1, n_patches):
            x_j = patches[j]['X']
            y_j = patches[j]['Y']
            
            # Check if both patches are within the grid
            if (1 <= x_i <= grid_size and 1 <= y_i <= grid_size and 
                1 <= x_j <= grid_size and 1 <= y_j <= grid_size):
                # Check if patches are direct neighbors along X or Y axis only
                if (abs(x_i - x_j) == 1 and y_i == y_j) or (abs(y_i - y_j) == 1 and x_i == x_j):
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
    output_path = '/Users/wolfram-sb/Projects/CDMetaPOP/helper_scripts/patch_connectivity_matrix.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write header
    patch_ids = [str(p['PatchID']) for p in patches]
    with open(output_path, 'w') as f:
        f.write(','.join([''] + patch_ids) + '\n')
        for i, row in enumerate(matrix):
            f.write(f"{patches[i]['PatchID']}," + ','.join(map(str, row)) + '\n')
    
    print(f"Matrix saved to: {output_path}")
    print(f"Matrix shape: {n_patches}x{n_patches}")
    print(f"Number of non-zero entries: {non_zero_count}")
    
    # Print summary of patches in the grid
    grid_patches = [p for p in patches if 1 <= p['X'] <= grid_size and 1 <= p['Y'] <= grid_size]
    print(f"\nPatches with X and Y in [1,{grid_size}]: {len(grid_patches)}")
    print("PatchID  X  Y")
    for p in grid_patches:
        print(f"{p['PatchID']:8} {p['X']:2} {p['Y']:2}")

if __name__ == '__main__':
    main()
