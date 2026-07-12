import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys


def get_largest_ind_file(directory):
    """
    Find the ind*.csv file with the largest number in its filename.
    Returns the path to the file or None if not found.
    """
    all_csv_files = [f for f in glob.glob(os.path.join(directory, 'ind*.csv')) if 'Sample' not in f]
    if not all_csv_files:
        return None
    
    file_numbers = []
    for f in all_csv_files:
        basename = os.path.basename(f)
        if basename.startswith('ind'):
            num_part = basename[3:].split('.')[0]
            file_numbers.append(int(num_part) if num_part.isdigit() else -1)
        else:
            file_numbers.append(-1)
    
    paired = list(zip(all_csv_files, file_numbers))
    paired.sort(key=lambda x: x[1])
    return paired[-1][0]


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


def create_plot(patch_data, title_prefix, output_file):
    """
    Create and save a two-panel plot for heterozygosity and population size.
    
    Args:
        patch_data: DataFrame with columns [PatchID, XCOORD, YCOORD, PopulationSize, Heterozygosity]
        title_prefix: String to use in plot titles
        output_file: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: Heterozygosity
    sc1 = ax1.scatter(
        x=patch_data['XCOORD'],
        y=patch_data['YCOORD'],
        c=patch_data['Heterozygosity'],
        cmap='viridis',
        s=200,
        edgecolors='black',
        alpha=0.7,
        vmin=0,
        vmax=1
    )
    ax1.set_xlabel('XCOORD')
    ax1.set_ylabel('YCOORD')
    ax1.set_title(f'Heterozygosity by Patch - {title_prefix}')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label='Heterozygosity')
    
    # Subplot 2: Population Size
    sc2 = ax2.scatter(
        x=patch_data['XCOORD'],
        y=patch_data['YCOORD'],
        c=patch_data['PopulationSize'],
        cmap='plasma',
        s=200,
        edgecolors='black',
        alpha=0.7
    )
    ax2.set_xlabel('XCOORD')
    ax2.set_ylabel('YCOORD')
    ax2.set_title(f'Population Size by Patch - {title_prefix}')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label='Population Size')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_heterozygosity(input_dir, output_dir):
    """
    Analyze heterozygosity and population size from ind*.csv files.
    
    Args:
        input_dir: Directory containing ind*.csv files
        output_dir: Directory to save plots
    
    Returns:
        A DataFrame with columns [PatchID, XCOORD, YCOORD, PopulationSize, Heterozygosity]
        representing the matrix of minimum heterozygosity and population size per patch.
    """
    csv_file = get_largest_ind_file(input_dir)
    
    if csv_file is None:
        print(f"No files processed. Check that {input_dir} contains ind*.csv files.")
        return None
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Compute heterozygosity per PatchID: 1 - sum(p_i^2) where p_i = freq of each genotype pattern
    #genotype_cols = ['L0A0', 'L0A1', 'L1A0', 'L1A1']
    genotype_cols = ['L0A0', 'L0A1']
    
    def compute_heterozygosity(group):
        total = len(group)
        # Count each unique genotype pattern
        pattern_counts = group.groupby(genotype_cols).size()
        # print(pattern_counts/total)
        # p_i = count_i / total, sum(p_i^2)
        sum_p_squared = (pattern_counts / total).pow(2).sum()
        print(total,sum_p_squared)
        #print(sum_p_squared)
        return 1 - sum_p_squared
    
    patch_data = df.groupby('PatchID').agg(
        XCOORD=('XCOORD', 'first'),
        YCOORD=('YCOORD', 'first'),
        PopulationSize=('ID', 'count'),
        Heterozygosity=('ID', lambda x: compute_heterozygosity(df.loc[x.index]))
    ).reset_index()
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = format_base_name(csv_file)
    input_dir_last = os.path.basename(input_dir)
    output_file = os.path.join(output_dir, f'{base_name}_{input_dir_last}.png')
    
    create_plot(patch_data, os.path.basename(csv_file), output_file)
    
    # Print minimum heterozygosity and mean population size for this file
    min_heterozygosity = patch_data['Heterozygosity'].min()
    avg_population_size = patch_data['PopulationSize'].mean()
    print(f"Saved plot: {output_file}")
    print(f"Minimum heterozygosity: {min_heterozygosity:.4f}")
    print(f"Average population size: {avg_population_size:.1f}")
    
    print(f"Processed 1 file.")
    return patch_data

# Get scenario from command line argument
if len(sys.argv) < 2:
    print("Usage: python analysis_ai.py scenario")
    sys.exit(1)

scenario = sys.argv[1]
pattern = os.path.join('..', 'output', scenario, 'raw', '*', 'run0batch0mc*species0')
print(pattern)
run_dirs = glob.glob(pattern)

output_dir = os.path.join('..', 'output', scenario, 'analysed')
print(output_dir)

if not run_dirs:
    print("No directories found matching pattern.")
    sys.exit(1)

all_patch_data = []

for input_dir in run_dirs:
    result = analyze_heterozygosity(input_dir, output_dir)
    if result is not None:
        all_patch_data.append(result)

# Create combined plot across all input directories
if all_patch_data:
    combined_df = pd.concat(all_patch_data, ignore_index=True)
    
    # Find the largest ind file across all directories for consistent naming
    all_ind_files = []
    for d in run_dirs:
        csv_file = get_largest_ind_file(d)
        if csv_file:
            all_ind_files.append(csv_file)
    
    if all_ind_files:
        # Sort by the numeric part of the filename
        all_ind_files.sort(key=lambda f: int(os.path.basename(f)[3:].split('.')[0]) if os.path.basename(f).startswith('ind') else -1)
        base_name = format_base_name(all_ind_files[-1])
    else:
        base_name = 'combined'
    
    # Group by PatchID and compute min for Heterozygosity, mean for PopulationSize
    combined_patch_data = combined_df.groupby('PatchID').agg(
        XCOORD=('XCOORD', 'first'),
        YCOORD=('YCOORD', 'first'),
        PopulationSize=('PopulationSize', 'mean'),
        Heterozygosity=('Heterozygosity', 'min')
    ).reset_index()
    
    output_file = os.path.join(output_dir, f'{base_name}_minimum.png')
    create_plot(combined_patch_data, f'All Runs ({scenario})', output_file)
    
    print(f"\nSaved combined plot: {output_file}")
    print(f"Overall minimum heterozygosity: {combined_patch_data['Heterozygosity'].min():.4f}")
    print(f"Overall average population size: {combined_patch_data['PopulationSize'].mean():.1f}")