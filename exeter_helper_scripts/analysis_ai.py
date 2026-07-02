import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

# Directory containing the CSV files
input_dir = 'output/1782241366/run0batch0mc0species0'
output_dir = 'output/plots'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all ind*.csv files
csv_files = [f for f in glob.glob(os.path.join(input_dir, 'ind*.csv')) if 'Sample' not in f]

for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Compute heterozygosity per PatchID: 1 - sum(p_i^2) where p_i = freq of each genotype pattern
    #genotype_cols = ['L0A0', 'L0A1', 'L1A0', 'L1A1']
    genotype_cols = ['L0A0', 'L0A1']
    
    def compute_heterozygosity(group):
        total = len(group)
        # Count each unique genotype pattern
        pattern_counts = group.groupby(genotype_cols).size()
        # p_i = count_i / total, sum(p_i^2)
        sum_p_squared = (pattern_counts / total).pow(2).sum()
        return 1 - sum_p_squared
    
    patch_data = df.groupby('PatchID').agg(
        XCOORD=('XCOORD', 'first'),
        YCOORD=('YCOORD', 'first'),
        PopulationSize=('ID', 'count'),
        Heterozygosity=('ID', lambda x: compute_heterozygosity(df.loc[x.index]))
    ).reset_index()
    
    # Create figure with two subplots: heterozygosity and population size
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
    ax1.set_title(f'Heterozygosity by Patch - {os.path.basename(csv_file)}')
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
    ax2.set_title(f'Population Size by Patch - {os.path.basename(csv_file)}')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label='Population Size')
    
    plt.tight_layout()
    
    # Save the plot with name corresponding to input file
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    # Pad the number in filename with leading zeros for proper sorting
    if base_name.startswith('ind'):
        num_part = base_name[3:]
        if num_part.isdigit():
            base_name = 'ind' + num_part.zfill(6)
    output_file = os.path.join(output_dir, f'{base_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print average heterozygosity and population size for this file
    avg_heterozygosity = patch_data['Heterozygosity'].mean()
    avg_population_size = patch_data['PopulationSize'].mean()
    print(f"Saved plot: {output_file}")
    print(f"Average heterozygosity: {avg_heterozygosity:.4f}")
    print(f"Average population size: {avg_population_size:.1f}")

print(f"Processed {len(csv_files)} files.")
