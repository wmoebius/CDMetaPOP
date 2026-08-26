import pandas as pd
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gc
from scipy.optimize import curve_fit


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Calculate population heterozygosity and population size for each patch across Monte-Carlo runs."
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing the Monte-Carlo run directories"
)

parser.add_argument(
    "--no-heatmaps",
    action="store_true",
    help="Do not create heatmaps"
)

args = parser.parse_args()


# ============================================================
# CHECK DIRECTORY
# ============================================================

output_dir = Path(args.d)

if not output_dir.exists():
    raise FileNotFoundError(
        f"Directory does not exist: {output_dir}"
    )

if not output_dir.is_dir():
    raise NotADirectoryError(
        f"Not a directory: {output_dir}"
    )


# ============================================================
# FIND MONTE-CARLO RUN DIRECTORIES
# ============================================================

run_directories = []

for directory in output_dir.iterdir():

    if not directory.is_dir():
        continue

    match = re.fullmatch(
        r"run(\d+)batch(\d+)mc(\d+)species(\d+)",
        directory.name
    )

    if match:

        run_number = int(match.group(1))
        batch_number = int(match.group(2))
        mc_number = int(match.group(3))
        species_number = int(match.group(4))

        run_directories.append(
            (
                mc_number,
                directory
            )
        )


run_directories.sort(
    key=lambda x: x[0]
)


if len(run_directories) == 0:

    raise FileNotFoundError(
        f"No Monte-Carlo run directories found in "
        f"{output_dir}"
    )


print()

print(
    f"Found {len(run_directories)} Monte-Carlo runs."
)

for mc_number, directory in run_directories:

    print(
        f"  MC {mc_number}: {directory.name}"
    )


# ============================================================
# FIND IND FILES
# ============================================================

def find_ind_files(run_directory):

    ind_files = []

    for file in run_directory.glob("ind*.csv"):

        match = re.fullmatch(
            r"ind(\d+)\.csv",
            file.name
        )

        if match:

            generation = int(match.group(1))

            ind_files.append(
                (
                    generation,
                    file
                )
            )

    ind_files.sort(
        key=lambda x: x[0]
    )

    if len(ind_files) == 0:

        raise FileNotFoundError(
            f"No ind<number>.csv files found in "
            f"{run_directory}"
        )

    return ind_files


# ============================================================
# CALCULATE INDIVIDUAL HETEROZYGOSITY
# ============================================================

def calculate_individual_heterozygosity(df):

    # --------------------------------------------------------
    # Locus 0
    #
    # Heterozygous if:
    # L0A0 = 1
    # L0A1 = 1
    # --------------------------------------------------------

    heterozygous_L0 = (
        (df["L0A0"] == 1) &
        (df["L0A1"] == 1)
    )

    # --------------------------------------------------------
    # Locus 1
    #
    # Heterozygous if:
    # L1A0 = 1
    # L1A1 = 1
    # --------------------------------------------------------

    heterozygous_L1 = (
        (df["L1A0"] == 1) &
        (df["L1A1"] == 1)
    )

    # --------------------------------------------------------
    # Individual heterozygosity
    #
    # Each heterozygous locus contributes 1.
    # Divide by number of loci.
    # --------------------------------------------------------

    df["Heterozygosity"] = (
        heterozygous_L0.astype(float) +
        heterozygous_L1.astype(float)
    ) / 2.0

    return df


# ============================================================
# FIND ALL PATCHES IN A RUN
# ============================================================

def find_all_patches(ind_data, generations):

    all_patches = set()

    for generation in generations:

        df = ind_data[generation]

        for patch in df["PatchID"].unique():

            all_patches.add(patch)

    return sorted(all_patches)


# ============================================================
# PLOT HEATMAP
# ============================================================

def plot_heatmap(
    matrix,
    generations,
    all_patches,
    save_directory,
    title,
    filename,
    colourbar_label,
    vmin=0,
    vmax=None,
    decimals=2
):

    fig_width = max(
        8,
        len(all_patches) * 1.0
    )

    fig_height = max(
        6,
        len(generations) * 0.5
    )

    fig, ax = plt.subplots(
        figsize=(
            fig_width,
            fig_height
        )
    )

    # ========================================================
    # COLOUR = MATRIX VALUE
    # ========================================================

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=vmin,
        vmax=vmax
    )

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlabel(
        "Patch ID"
    )

    ax.set_ylabel(
        "Generation"
    )

    ax.set_title(
        title
    )

    ax.set_xticks(
        range(
            len(all_patches)
        )
    )

    ax.set_xticklabels(
        all_patches
    )

    ax.set_yticks(
        range(
            len(generations)
        )
    )

    ax.set_yticklabels(
        generations
    )

    # ========================================================
    # CELL LABELS
    # ========================================================

    for row_number in range(
        len(generations)
    ):

        for column_number in range(
            len(all_patches)
        ):

            value = (
                matrix.iloc[
                    row_number,
                    column_number
                ]
            )

            if not pd.isna(value):

                if decimals == 0:

                    label = f"{value:.0f}"

                else:

                    label = f"{value:.{decimals}f}"

                ax.text(
                    column_number,
                    row_number,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8
                )

    # ========================================================
    # COLOURBAR
    # ========================================================

    colourbar = fig.colorbar(
        image,
        ax=ax
    )

    colourbar.set_label(
        colourbar_label
    )

    if vmax == 1:

        colourbar.set_ticks(
            [
                0,
                0.25,
                0.5,
                0.75,
                1.0
            ]
        )

    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    output_file = (
        save_directory /
        f"{filename}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Heatmap written to:"
    )

    print(
        f"    {output_file}"
    )


# ============================================================
# PLOT LANDSCAPE-WIDE AVERAGE HETEROZYGOSITY
# ============================================================

def plot_landscape_average(
    heterozygosity_matrix,
    generations,
    save_directory,
    title,
    filename,
    graphtype = "Heterozygosity",
    log=False,
    annotations=False
):

    # --------------------------------------------------------
    # Calculate the mean heterozygosity across patches for
    # each generation.
    #
    # Empty patches contain NaN heterozygosity and are
    # therefore excluded from the mean.
    # --------------------------------------------------------

    landscape_average = (
        heterozygosity_matrix
        .mean(
            axis=1,
            skipna=True
        )
    )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        generations,
        landscape_average.values,
        linewidth=2
    )

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlabel(
        "Generation"
    )

    ax.set_ylabel(
        "Mean heterozygosity"
    )

    ax.set_title(
        title
    )

    if log:
        ax.set_ylim(
            1e-1,
            1
        )

    else:
        ax.set_ylim(
            0,
            150
        )

    ax.set_xlim(
        generations[0],
        generations[-1]
    )

    ax.grid(
        alpha=0.3
    )

    if log:
        ax.set_yscale('log')

    if graphtype == "Heterozygosity":
        if annotations:
            ##The expectation
            x = [generations[int(len(generations)/3)],
                generations[int(len(generations)/3*2)]]
            y = [0.6* np.exp(-0.5 * x[0] / 100), 0.6* np.exp(-0.5 * x[1] / 100)]


            ax.plot(
                x,
                y,
                linewidth=2
            )

            plt.text(x[0],y[0],"Theory: Disconnected")

            y = [0.6 * np.exp(-0.5 * x[0] / (20*100)), 0.6* np.exp(-0.5 * x[1] / (20*100))]

            ax.plot(
                x,
                y,
                linewidth=2
            )

            plt.text(x[0],y[0],"Theory: Fully Connected")



            #The results
            def fitfunc(x,a,b):
                return b * np.exp(-x /a)

            popt,pcov = curve_fit(fitfunc,
                                np.asarray(generations),
                                landscape_average.values)

            a = popt[0]
            b = popt[1]

            print("Fitting params:",a,b)

            plt.plot(generations,fitfunc(np.asarray(generations),a,b),
                    linestyle='dashed')

            plt.text(generations[0],
                    landscape_average.values[0],
                    r"$\lambda = 1 /(2 \times %0.3f)$"%(a/2))

    if graphtype == "Number":
        numberdata = np.asarray(landscape_average.values)
        Mean = np.mean(numberdata[len(numberdata)//2:])

        plt.text(generations[int(len(generations)/2)],
                 50,
                 "Mean = %0.3f"%(Mean))

    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    output_file = (
        save_directory /
        f"{filename}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Landscape-average heterozygosity plot written to:"
    )

    print(
        f"    {output_file}"
    )


# ============================================================
# PROCESS ONE MONTE-CARLO RUN
# ============================================================

def process_run(run_directory):

    print()
    print("================================================")
    print(
        f"Processing: {run_directory.name}"
    )
    print("================================================")

    # ========================================================
    # FIND IND FILES
    # ========================================================

    ind_files = find_ind_files(
        run_directory
    )

    generations = []

    for generation, file in ind_files:

        generations.append(
            generation
        )

    print(
        f"Generations: {generations[0]} -> "
        f"{generations[-1]}"
    )

    # ========================================================
    # READ IND FILES
    # ========================================================

    ind_data = {}

    for generation, file in ind_files:

        print(
            f"  Reading generation {generation}"
        )

        df = pd.read_csv(file)

        # ----------------------------------------------------
        # Calculate individual heterozygosity
        # ----------------------------------------------------

        df = calculate_individual_heterozygosity(
            df
        )

        ind_data[generation] = df

    # ========================================================
    # FIND PATCHES
    # ========================================================

    all_patches = find_all_patches(
        ind_data,
        generations
    )

    print(
        f"  Found {len(all_patches)} patches."
    )

    # ========================================================
    # CALCULATE PATCH HETEROZYGOSITY AND N
    # ========================================================

    results = []

    for generation in generations:

        df = ind_data[generation]

        print(
            f"  Calculating generation {generation}"
        )

        # ----------------------------------------------------
        # Calculate mean heterozygosity for each patch
        # ----------------------------------------------------

        patch_means = (
            df.groupby("PatchID")["Heterozygosity"]
            .mean()
        )

        for patch in all_patches:

            if patch in patch_means.index:

                heterozygosity = (
                    patch_means.loc[patch]
                )

                number_individuals = (
                    df["PatchID"] == patch
                ).sum()

            else:

                # No individuals in this patch.
                #
                # Heterozygosity is undefined.
                # N is zero.

                heterozygosity = np.nan

                number_individuals = 0

            results.append(
                {
                    "Generation": generation,
                    "PatchID": patch,
                    "Heterozygosity": heterozygosity,
                    "N": number_individuals
                }
            )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # CREATE HETEROZYGOSITY MATRIX
    # ========================================================

    heterozygosity_matrix = (
        results_df
        .pivot(
            index="Generation",
            columns="PatchID",
            values="Heterozygosity"
        )
        .reindex(
            index=generations,
            columns=all_patches
        )
    )

    # ========================================================
    # CREATE N MATRIX
    # ========================================================

    n_matrix = (
        results_df
        .pivot(
            index="Generation",
            columns="PatchID",
            values="N"
        )
        .reindex(
            index=generations,
            columns=all_patches
        )
        .fillna(0)
    )

    # ========================================================
    # PLOT INDIVIDUAL RUN HEATMAPS
    # ========================================================

    if not args.no_heatmaps:

        print(
            "  Creating run heterozygosity heatmap..."
        )

        plot_heatmap(
            heterozygosity_matrix,
            generations,
            all_patches,
            run_directory,
            f"Heterozygosity\n{run_directory.name}",
            "heterozygosity_heatmap",
            colourbar_label="Mean heterozygosity",
            vmin=0,
            vmax=1,
            decimals=2
        )

        print(
            "  Creating run population-size heatmap..."
        )

        plot_heatmap(
            n_matrix,
            generations,
            all_patches,
            run_directory,
            f"Number of individuals\n{run_directory.name}",
            "N_heatmap",
            colourbar_label="Number of individuals",
            vmin=0,
            vmax=None,
            decimals=0
        )

    # ----------------------------------------------------
    # Separate landscape-average plot
    # ----------------------------------------------------

    print(
        "  Creating run landscape-average "
        "heterozygosity plot..."
    )

    plot_landscape_average(
        heterozygosity_matrix,
        generations,
        run_directory,
        f"Landscape-wide average heterozygosity\n"
        f"{run_directory.name}",
        "heterozygosity_landscape_average",
        log=True
    )

    print(
        "  Creating run landscape-average "
        "number of individuals..."
    )

    plot_landscape_average(
        n_matrix,
        generations,
        run_directory,
        f"Landscape-wide average population size\n"
        f"{run_directory.name}",
        "N_landscape_average",
        graphtype="Number"
    )


    # ========================================================
    # SAVE RUN DATA
    # ========================================================

    run_output = (
        run_directory /
        "heterozygosity.csv"
    )

    results_df.to_csv(
        run_output,
        index=False
    )

    print(
        f"  Heterozygosity data written to:"
    )

    print(
        f"    {run_output}"
    )

    # ========================================================
    # SAVE N DATA
    # ========================================================

    n_output = (
        run_directory /
        "N.csv"
    )

    n_matrix.to_csv(
        n_output
    )

    print(
        f"  Population-size data written to:"
    )

    print(
        f"    {n_output}"
    )

    # ========================================================
    # MEMORY CLEANUP
    # ========================================================

    del ind_data

    gc.collect()

    return (
        heterozygosity_matrix.copy(),
        n_matrix.copy(),
        generations,
        all_patches
    )


# ============================================================
# PROCESS ALL MONTE-CARLO RUNS
# ============================================================

all_heterozygosity_matrices = []
all_n_matrices = []

common_generations = None
common_patches = None


for mc_number, run_directory in run_directories:

    (
        heterozygosity_matrix,
        n_matrix,
        generations,
        patches
    ) = process_run(
        run_directory
    )

    # ========================================================
    # MAKE SURE ALL RUNS HAVE SAME DIMENSIONS
    # ========================================================

    if common_generations is None:

        common_generations = generations
        common_patches = patches

    else:

        if generations != common_generations:

            raise ValueError(
                f"Generation mismatch between "
                f"Monte-Carlo runs.\n"
                f"Expected: {common_generations}\n"
                f"Found: {generations}\n"
                f"Problem run: {run_directory}"
            )

        if patches != common_patches:

            raise ValueError(
                f"Patch mismatch between "
                f"Monte-Carlo runs.\n"
                f"Expected: {common_patches}\n"
                f"Found: {patches}\n"
                f"Problem run: {run_directory}"
            )

    # ========================================================
    # STORE SMALL MATRICES
    # ========================================================

    all_heterozygosity_matrices.append(
        heterozygosity_matrix
    )

    all_n_matrices.append(
        n_matrix
    )

    del heterozygosity_matrix
    del n_matrix

    gc.collect()


# ============================================================
# COMBINE MONTE-CARLO RUNS
# ============================================================

print()
print("================================================")
print("Combining Monte-Carlo runs...")
print("================================================")

number_runs = len(
    all_heterozygosity_matrices
)


# ============================================================
# SUMMED HETEROZYGOSITY
# ============================================================

summed_heterozygosity_matrix = (
    pd.concat(
        all_heterozygosity_matrices,
        axis=0
    )
    .groupby(level=0)
    .sum(
        skipna=True
    )
)


# ============================================================
# AVERAGE HETEROZYGOSITY ACROSS MC RUNS
# ============================================================

average_heterozygosity_matrix = (
    pd.concat(
        all_heterozygosity_matrices,
        axis=0
    )
    .groupby(level=0)
    .mean(
        skipna=True
    )
    .reindex(
        index=common_generations,
        columns=common_patches
    )
)


# ============================================================
# NUMBER OF RUNS CONTRIBUTING TO EACH HETEROZYGOSITY CELL
# ============================================================

number_contributing_runs = (
    pd.concat(
        [
            matrix.notna().astype(int)
            for matrix in all_heterozygosity_matrices
        ],
        axis=0
    )
    .groupby(level=0)
    .sum()
    .reindex(
        index=common_generations,
        columns=common_patches
    )
)


# ============================================================
# AVERAGE NUMBER OF INDIVIDUALS ACROSS MC RUNS
# ============================================================

#
# Empty patches contribute N = 0.
# Therefore every Monte-Carlo run contributes to the average.
#

average_n_matrix = (
    pd.concat(
        all_n_matrices,
        axis=0
    )
    .groupby(level=0)
    .mean()
    .reindex(
        index=common_generations,
        columns=common_patches
    )
)


# ============================================================
# SUMMED NUMBER OF INDIVIDUALS ACROSS MC RUNS
# ============================================================

summed_n_matrix = (
    pd.concat(
        all_n_matrices,
        axis=0
    )
    .groupby(level=0)
    .sum()
    .reindex(
        index=common_generations,
        columns=common_patches
    )
)


# ============================================================
# AVERAGE NUMBER OF INDIVIDUALS ACROSS MC RUNS
# ============================================================

average_n_matrix = (
    pd.concat(
        all_n_matrices,
        axis=0
    )
    .groupby(level=0)
    .mean()
    .reindex(
        index=common_generations,
        columns=common_patches
    )
)


# ============================================================
# SAVE SUMMED HETEROZYGOSITY
# ============================================================

summed_output = (
    output_dir /
    "heterozygosity_SUM.csv"
)

summed_heterozygosity_matrix.to_csv(
    summed_output
)


# ============================================================
# SAVE AVERAGED HETEROZYGOSITY
# ============================================================

average_output = (
    output_dir /
    "heterozygosity_AVERAGE.csv"
)

average_heterozygosity_matrix.to_csv(
    average_output
)


# ============================================================
# SAVE NUMBER OF CONTRIBUTING RUNS
# ============================================================

runs_output = (
    output_dir /
    "heterozygosity_NUMBER_OF_RUNS.csv"
)

number_contributing_runs.to_csv(
    runs_output
)


# ============================================================
# SAVE AVERAGE POPULATION SIZE
# ============================================================

average_n_output = (
    output_dir /
    "N_AVERAGE.csv"
)

average_n_matrix.to_csv(
    average_n_output
)


# ============================================================
# SAVE SUMMED POPULATION SIZE
# ============================================================

summed_n_output = (
    output_dir /
    "N_SUM.csv"
)

summed_n_matrix.to_csv(
    summed_n_output
)


# ============================================================
# PRINT OUTPUT FILES
# ============================================================

print()

print(
    f"Summed heterozygosity data written to:"
)

print(
    f"  {summed_output}"
)

print()

print(
    f"Average heterozygosity data written to:"
)

print(
    f"  {average_output}"
)

print()

print(
    f"Number-of-contributing-runs data written to:"
)

print(
    f"  {runs_output}"
)

print()

print(
    f"Average population-size data written to:"
)

print(
    f"  {average_n_output}"
)

print()

print(
    f"Summed population-size data written to:"
)

print(
    f"  {summed_n_output}"
)


# ============================================================
# PLOT AVERAGED HETEROZYGOSITY HEATMAP
# ============================================================

if not args.no_heatmaps:

    print()
    print(
        "Creating averaged heterozygosity heatmap..."
    )

    plot_heatmap(
        average_heterozygosity_matrix,
        common_generations,
        common_patches,
        output_dir,
        f"AVERAGED population heterozygosity\n"
        f"{number_runs} Monte-Carlo runs",
        "heterozygosity_AVERAGE",
        colourbar_label="Mean heterozygosity",
        vmin=0,
        vmax=1,
        decimals=2
    )


# ============================================================
# PLOT AVERAGED POPULATION-SIZE HEATMAP
# ============================================================

if not args.no_heatmaps:

    print()
    print(
        "Creating averaged population-size heatmap..."
    )

    plot_heatmap(
        average_n_matrix,
        common_generations,
        common_patches,
        output_dir,
        f"AVERAGED number of individuals\n"
        f"{number_runs} Monte-Carlo runs",
        "N_AVERAGE",
        colourbar_label="Mean number of individuals",
        vmin=0,
        vmax=None,
        decimals=1
    )


# ============================================================
# PLOT AVERAGED LANDSCAPE-WIDE HETEROZYGOSITY
# ============================================================

#if not args.no_heatmaps:

print()
print(
    "Creating averaged landscape-wide "
    "heterozygosity plot..."
)

plot_landscape_average(
    average_heterozygosity_matrix,
    common_generations,
    output_dir,
    f"Landscape-wide average heterozygosity\n"
    f"{number_runs} Monte-Carlo runs",
    "heterozygosity_AVERAGE_landscape_average",
    log=True,
    annotations=True
)

plot_landscape_average(
    average_n_matrix,
    common_generations,
    output_dir,
    f"Landscape-wide average population size\n"
    f"{number_runs} Monte-Carlo runs",
    "N_landscape_average",
    graphtype="Number"
)

# ============================================================
# CLEAN UP
# ============================================================

del all_heterozygosity_matrices
del all_n_matrices

gc.collect()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("================================================")
print("Finished.")
print("================================================")

print(
    f"Monte-Carlo runs processed: {number_runs}"
)

print(
    f"Generations: "
    f"{common_generations[0]} -> "
    f"{common_generations[-1]}"
)

print(
    f"Number of patches: {len(common_patches)}"
)

print()

print(
    "Individual heterozygosity:"
)

print(
    "  0.0 = homozygous at both loci"
)

print(
    "  0.5 = heterozygous at one locus"
)

print(
    "  1.0 = heterozygous at both loci"
)

print()

print(
    "Population size:"
)

print(
    "  N = number of individuals in each patch"
)

print(
    "  Empty patches contribute N = 0"
)

print()

print(
    "Landscape-wide heterozygosity:"
)

print(
    "  Calculated as the mean of occupied-patch "
    "heterozygosity at each generation."
)

print()

print(
    "Empty patches are excluded from Monte-Carlo "
    "heterozygosity averages."
)

print(
    "Empty patches are INCLUDED as N = 0 in "
    "population-size averages."
)