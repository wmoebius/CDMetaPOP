import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys


def get_all_ind_files(directory):
    """
    Find all ind*.csv files in the directory, sorted by time number.
    Returns a list of (filepath, time_number) tuples.
    """
    all_csv_files = [f for f in glob.glob(os.path.join(directory, 'ind*.csv')) if 'Sample' not in f]
    if not all_csv_files:
        return []
    
    file_data = []
    for f in all_csv_files:
        basename = os.path.basename(f)
        if basename.startswith('ind'):
            num_part = basename[3:].split('.')[0]
            if num_part.isdigit():
                file_data.append((f, int(num_part)))
    
    # Sort by time number
    file_data.sort(key=lambda x: x[1])
    return file_data


def format_base_name(csv_file):
    """
    Extract and format the base name from a CSV file path.
    Pads the number with leading zeros for proper sorting.
    """
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    if base_name.startswith('ind'):
        num_part = base_name[3:]
        if num_part.isdigit():
            base_name = 'ind' + num_part.zfill(6)
    return base_name


def process_input_dir(input_dir, output_dir):
    """
    Process a single input directory containing ind*.csv files.
    Creates and saves a movement trajectories plot.
    
    Args:
        input_dir: Directory containing ind*.csv files
        output_dir: Directory to save outputs
    
    Returns:
        None
    """
    ind_files = get_all_ind_files(input_dir)
    
    if not ind_files:
        print(f"No files processed. Check that {input_dir} contains ind*.csv files.")
        return None
    
    all_dfs = []
    for csv_file, time_num in ind_files:
        # Read the CSV file with only needed columns
        df = pd.read_csv(csv_file, usecols=['ID', 'XCOORD', 'YCOORD'])
        # Add time column
        df['Time'] = time_num
        all_dfs.append(df)
    
    # Concatenate all time points from this folder
    combined_df = pd.concat(all_dfs, ignore_index=True)[['ID', 'Time', 'XCOORD', 'YCOORD']]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the last (largest time) file for naming
    last_file = ind_files[-1][0]
    base_name = format_base_name(last_file)
    input_dir_last = os.path.basename(input_dir)
    
    # Create movement visualization
    # Remove duplicate (x, y) for scatterplot
    unique_positions = combined_df.drop_duplicates(subset=['XCOORD', 'YCOORD'])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Scatterplot of all unique (x, y)
    ax.scatter(unique_positions['XCOORD'], unique_positions['YCOORD'],
               color='tab:blue', alpha=0.7, s=40, label='Positions')
    
    # Filter IDs that move (have different positions across time points)
    grouped = combined_df.groupby('ID')
    for id, group in grouped:
        # print(group)
        # Check each ID occurs once per Time
        time_counts = group['Time'].value_counts()
        if not all(time_counts == 1):
            print(f"Warning: ID {id} has duplicate entries for some Time values")
        
        # Check if ID moves (has at least 2 different positions)
        unique_coords = group[['XCOORD', 'YCOORD']].drop_duplicates()
        if len(unique_coords) < 2:
            continue
        
        # Sort by time
        group = group.sort_values('Time')
        print(group)
        
        # Plot trajectory as line connecting positions over time
        ax.plot(group['XCOORD'], group['YCOORD'], 
                marker='o', markersize=4, linewidth=1, label=f'ID {id}')
    
    ax.set_xlabel('XCOORD')
    ax.set_ylabel('YCOORD')
    ax.set_title(f'Movement Trajectories - {input_dir_last}')
    
    output_file = os.path.join(output_dir, f'movement_trajectories_{input_dir_last}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Processed {len(ind_files)} files from {input_dir}")
    print(f"Saved plot: {output_file}")


# Get scenario from command line argument
if len(sys.argv) < 2:
    print("Usage: python analysis_movement_ai.py scenario")
    sys.exit(1)

scenario = sys.argv[1]
pattern = os.path.join('..', 'output', scenario, 'raw', '*', 'run0batch0mc*species0')
print(pattern)

output_dir = os.path.join('..', 'output', scenario, 'analysed')
print(output_dir)

all_patch_data = []

for input_dir in glob.glob(pattern):
    result = process_input_dir(input_dir, output_dir)