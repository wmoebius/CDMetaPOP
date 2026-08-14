import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys


def get_all_ind_files(directory):
    """
    Find all ind*.csv files in a directory and return them sorted by their numeric part.
    Returns a list of tuples (filepath, time_number).
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


def compute_heterozygosity(group, genotype_cols):
    """Compute heterozygosity for a group of individuals: 1 - sum(p_i^2)."""
    total = len(group)
    if total == 0:
        return 0.0
    pattern_counts = group.groupby(genotype_cols).size()
    sum_p_squared = (pattern_counts / total).pow(2).sum()
    return 1 - sum_p_squared


def compute_patch_stats(df, genotype_cols=['L0A0', 'L0A1', 'L1A0', 'L1A1']):
    """
    Compute heterozygosity and population size for each patch.
    
    Args:
        df: DataFrame containing individual data
        genotype_cols: List of column names representing genotype patterns
        
    Returns:
        DataFrame with columns [PatchID, XCOORD, YCOORD, PopulationSize, Heterozygosity]
    """
    def get_heterozygosity(group):
        return compute_heterozygosity(group, genotype_cols)
    
    patch_data = df.groupby('PatchID').agg(
        XCOORD=('XCOORD', 'first'),
        YCOORD=('YCOORD', 'first'),
        PopulationSize=('ID', 'count'),
        Heterozygosity=('ID', lambda x: get_heterozygosity(df.loc[x.index]))
    ).reset_index()
    
    return patch_data


def create_spatial_plot(patch_data, title_prefix, output_file):
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


def analyze_run(input_dir, genotype_cols=['L0A0', 'L0A1', 'L1A0', 'L1A1']):
    """
    Analyze heterozygosity for a single run directory.
    
    Args:
        input_dir: Directory containing ind*.csv files
        genotype_cols: List of genotype column names
        
    Returns:
        DataFrame with columns [Time, RunID, PatchID, XCOORD, YCOORD, PopulationSize, Heterozygosity]
    """
    ind_files = get_all_ind_files(input_dir)
    
    if not ind_files:
        print(f"  No files processed. Check that {input_dir} contains ind*.csv files.")
        return None
    
    input_dir_last = os.path.basename(input_dir)
    print(f"  Found {len(ind_files)} time points in {input_dir_last}")
    
    all_patch_data = []
    
    for csv_file, time_num in ind_files:
        df = pd.read_csv(csv_file)
        patch_data = compute_patch_stats(df, genotype_cols)
        
        for _, row in patch_data.iterrows():
            all_patch_data.append({
                'Time': time_num,
                'RunID': input_dir_last,
                'PatchID': row['PatchID'],
                'XCOORD': row['XCOORD'],
                'YCOORD': row['YCOORD'],
                'PopulationSize': row['PopulationSize'],
                'Heterozygosity': row['Heterozygosity']
            })
        
        avg_heterozygosity = patch_data['Heterozygosity'].mean()
        avg_population_size = patch_data['PopulationSize'].mean()
        print(f"    Time {time_num}: avg_heterozygosity={avg_heterozygosity:.4f}, "
              f"avg_population={avg_population_size:.1f}, patches={len(patch_data)}")
    
    result_df = pd.DataFrame(all_patch_data)
    print(f"  Processed {len(ind_files)} files.")
    return result_df


def main():
    # Get scenario from command line argument
    if len(sys.argv) < 2:
        print("Usage: python analysis_heterozygosity_ai.py scenario")
        sys.exit(1)
    
    scenario = sys.argv[1]
    pattern = os.path.join(scenario, 'outputs', 'raw', '*', 'run0batch0mc*species0')
    print(f"Searching for directories with pattern: {pattern}")
    run_dirs = glob.glob(pattern)
    
    output_dir = os.path.join(scenario, 'outputs', 'analysed')
    print(f"Output directory: {output_dir}")
    
    if not run_dirs:
        print("No directories found matching pattern.")
        sys.exit(1)
    
    # Process each run directory
    genotype_cols = ['L0A0', 'L0A1', 'L1A0', 'L1A1']
    all_run_data = []
    
    for input_dir in run_dirs:
        result_df = analyze_run(input_dir, genotype_cols)
        if result_df is not None:
            all_run_data.append(result_df)
    
    if not all_run_data:
        print("No data to process.")
        sys.exit(1)
    
    # Combine all data
    combined_df = pd.concat(all_run_data, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # 1. Spatial map plots for each time point (averaged across runs)
    # ============================================================
    all_times = sorted(combined_df['Time'].unique())
    
    for time_num in all_times:
        time_data = combined_df[combined_df['Time'] == time_num]
        
        # Group by PatchID and compute mean across runs
        combined_patch_data = time_data.groupby('PatchID').agg(
            XCOORD=('XCOORD', 'first'),
            YCOORD=('YCOORD', 'first'),
            PopulationSize=('PopulationSize', 'mean'),
            Heterozygosity=('Heterozygosity', 'mean')
        ).reset_index()
        
        base_name = f'ind{time_num:06d}'
        output_file = os.path.join(output_dir, f'{base_name}_average.png')
        create_spatial_plot(combined_patch_data, f'All Runs ({scenario}) - Time {time_num}', output_file)
        
        print(f"\nSaved spatial map for time {time_num}: {output_file}")
        print(f"  Average heterozygosity: {combined_patch_data['Heterozygosity'].mean():.4f}")
        print(f"  Average population size: {combined_patch_data['PopulationSize'].mean():.1f}")
    
    # ============================================================
    # 2. Spatial map plots per replicate per time point
    # ============================================================
    for run_id in combined_df['RunID'].unique():
        run_data = combined_df[combined_df['RunID'] == run_id]
        run_times = sorted(run_data['Time'].unique())
        for time_num in run_times:
            time_data = run_data[run_data['Time'] == time_num]
            patch_data = time_data[['PatchID', 'XCOORD', 'YCOORD', 'PopulationSize', 'Heterozygosity']].copy()
            base_name = f'ind{time_num:06d}'
            output_file = os.path.join(output_dir, f'{base_name}_run_{run_id}.png')
            create_spatial_plot(patch_data, f'{run_id} - Time {time_num}', output_file)
            print(f"  Saved replicate map for {run_id} time {time_num}: {output_file}")
    
    # ============================================================
    # 3. Time series: per-patch heterozygosity averaged across runs
    # ============================================================
    patch_time_series = combined_df.groupby(['PatchID', 'Time']).agg(
        AverageHeterozygosity=('Heterozygosity', 'mean')
    ).reset_index()
    
    if not patch_time_series.empty:
        output_file = os.path.join(output_dir, 'heterozygosity_per_patch_average_over_runs.png')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for patch_id in patch_time_series['PatchID'].unique():
            patch_data = patch_time_series[patch_time_series['PatchID'] == patch_id].sort_values('Time')
            ax.plot(patch_data['Time'], patch_data['AverageHeterozygosity'],
                    marker='o', linestyle='-', markersize=3)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Heterozygosity')
        ax.set_title('Per-Patch Heterozygosity (averaged across runs) over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved per-patch average time series: {output_file}")
    
    # ============================================================
    # 4. Per-patch time series: one plot per run showing all patches
    # ============================================================
    for run_id in combined_df['RunID'].unique():
        run_data = combined_df[combined_df['RunID'] == run_id]
        output_file = os.path.join(output_dir, f'heterozygosity_per_patch_{run_id}.png')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        unique_patches = run_data['PatchID'].unique()
        for patch_id in unique_patches:
            patch_data = run_data[run_data['PatchID'] == patch_id].sort_values('Time')
            ax.plot(patch_data['Time'], patch_data['Heterozygosity'],
                    marker='o', linestyle='-', markersize=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Heterozygosity')
        ax.set_title(f'Heterozygosity over Time per Patch - {run_id}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved per-patch plot: {output_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
