import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ============================================================================
# ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description="Calculate alpha, beta and gamma genetic diversity."
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing the Monte-Carlo run directories",
)

parser.add_argument(
    "--no-heatmaps",
    action="store_true",
    help="Do not create heatmaps",
)

parser.add_argument(
    "--no-plots",
    action="store_true",
    help="Do not create plots",
)

parser.add_argument(
    "--no-save",
    action="store_true",
    help="Do not create the collated Data.npz or CSV files",
)


# ============================================================================
# BASIC FUNCTIONS
# ============================================================================

def find_runs(output_dir):
    """Find Monte-Carlo run directories."""

    runs = []

    pattern = re.compile(
        r"run(\d+)batch(\d+)mc(\d+)species(\d+)"
    )

    for directory in output_dir.iterdir():

        if not directory.is_dir():
            continue

        match = pattern.fullmatch(directory.name)

        if match:
            mc = int(match.group(3))
            runs.append((mc, directory))

    runs.sort(key=lambda x: x[0])

    if not runs:
        raise FileNotFoundError(
            f"No Monte-Carlo run directories found in {output_dir}"
        )

    return runs


def find_ind_files(run_directory):
    """Find ind<number>.csv files and sort by generation."""

    files = []

    for file in run_directory.glob("ind*.csv"):

        match = re.fullmatch(r"ind(\d+)\.csv", file.name)

        if match:
            generation = int(match.group(1))
            files.append((generation, file))

    files.sort(key=lambda x: x[0])

    if not files:
        raise FileNotFoundError(
            f"No ind<number>.csv files found in {run_directory}"
        )

    return files


def find_loci(columns):
    """
    Find all loci represented by LxA0/LxA1 columns.

    Example:
        L0A0, L0A1, L1A0, L1A1
    gives:
        [0, 1]
    """

    loci = sorted({
        int(match.group(1))
        for column in columns
        if (match := re.fullmatch(r"L(\d+)A0", column))
    })

    if not loci:
        raise ValueError(
            "No locus columns found. Expected columns such as "
            "'L0A0', 'L0A1', 'L1A0', 'L1A1', ..."
        )

    for locus in loci:

        if (
            f"L{locus}A0" not in columns
            or f"L{locus}A1" not in columns
        ):
            raise ValueError(
                f"Incomplete allele data for locus {locus}. "
                f"Expected both L{locus}A0 and L{locus}A1."
            )

    return loci


# ============================================================================
# BETA DIVERSITY
# ============================================================================

def calculate_beta_diversity(
    allele_counts,
    population_sizes,
    generations,
    patches,
    locus_numbers,
):
    """
    Calculate the beta-diversity matrix for every generation.

    Beta diversity is the mean absolute difference in A1 allele
    frequency between two patches, averaged across loci:

        B_ij = mean_l |p_i,l - p_j,l|

    Therefore:

        B_ij = 0
            -> identical allele frequencies

        B_ij = 1
            -> maximally different allele frequencies

    Empty patches are represented by NaN.
    """

    n_generations = len(generations)
    n_patches = len(patches)
    n_loci = len(locus_numbers)

    beta_array = np.full(
        (n_generations, n_patches, n_patches),
        np.nan,
        dtype=np.float32,
    )

    patch_index = {
        patch: i
        for i, patch in enumerate(patches)
    }

    for generation_index, generation in enumerate(generations):

        # ---------------------------------------------------------------
        # Build allele-frequency matrix:
        #
        # rows    = patches
        # columns = loci
        # ---------------------------------------------------------------

        frequencies = np.full(
            (n_patches, n_loci),
            np.nan,
            dtype=np.float32,
        )

        for patch in patches:

            if patch not in population_sizes[generation]:
                continue

            n = population_sizes[generation][patch]

            if n == 0:
                continue

            row = patch_index[patch]

            for locus_index, locus in enumerate(locus_numbers):

                a1_count = allele_counts[
                    generation
                ][patch][locus]

                # Each diploid individual carries two alleles.
                frequencies[row, locus_index] = (
                    a1_count / (2.0 * n)
                )

        # ---------------------------------------------------------------
        # Pairwise beta diversity
        # ---------------------------------------------------------------

        for i in range(n_patches):

            if np.all(np.isnan(frequencies[i])):
                continue

            for j in range(i, n_patches):

                if np.all(np.isnan(frequencies[j])):
                    continue

                valid = (
                    np.isfinite(frequencies[i])
                    & np.isfinite(frequencies[j])
                )

                if not np.any(valid):
                    continue

                beta = np.mean(
                    np.abs(
                        frequencies[i, valid]
                        - frequencies[j, valid]
                    )
                )

                beta_array[
                    generation_index,
                    i,
                    j
                ] = beta

                beta_array[
                    generation_index,
                    j,
                    i
                ] = beta

    return beta_array


# ============================================================================
# PLOTTING
# ============================================================================

def plot_heatmap(
    matrix,
    generations,
    patches,
    directory,
    title,
    filename,
    colourbar_label,
    vmin=0,
    vmax=None,
    decimals=2,
):
    """Create and save a heatmap."""

    fig_width = max(8, len(patches))
    fig_height = max(6, len(generations) * 0.5)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Patch ID")
    ax.set_ylabel("Generation")
    ax.set_title(title)

    ax.set_xticks(range(len(patches)))
    ax.set_xticklabels(patches)

    ax.set_yticks(range(len(generations)))
    ax.set_yticklabels(generations)

    for row in range(len(generations)):

        for column in range(len(patches)):

            value = matrix.iloc[row, column]

            if pd.notna(value):

                if decimals == 0:
                    label = f"{value:.0f}"
                else:
                    label = f"{value:.{decimals}f}"

                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colourbar = fig.colorbar(image, ax=ax)
    colourbar.set_label(colourbar_label)

    if vmax == 1:
        colourbar.set_ticks(
            [0, 0.25, 0.5, 0.75, 1]
        )

    plt.tight_layout()

    output_file = directory / f"{filename}.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Plot written to:\n"
        f"    {output_file}"
    )


def plot_curves(
    generations,
    curves,
    average_curve,
    directory,
    filename,
    ylabel,
    title,
    log=False,
    repeat_number=-1,
):
    """Plot every MC curve and the MC average."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for curve in curves:

        ax.plot(
            generations,
            curve,
            alpha=0.35,
        )

    ax.plot(
        generations,
        average_curve,
        linewidth=2,
        color="k",
        label="Average",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xlim(
        generations[0],
        generations[-1],
    )

    if log:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_file = directory / f"{filename}.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    if repeat_number != -1:

        output_file = (
            directory
            / "../../../../"
            / f"{filename}_Landscape_{repeat_number}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()

    print(
        f"  Plot written to:\n"
        f"    {output_file}"
    )


def fit_heterozygosity_curve(
    generations,
    curve,
    directory,
    title,
    filename,
    repeat_number=-1,
):
    """Fit 0.5 * exp(-x / (2a)) to the averaged curve."""

    generations = np.asarray(
        generations,
        dtype=float,
    )

    curve = np.asarray(
        curve,
        dtype=float,
    )

    def fit_function(x, a):
        return 0.5 * np.exp(-x / (2 * a))

    valid = (
        np.isfinite(generations)
        & np.isfinite(curve)
        & (curve > 0)
    )

    popt, pcov = curve_fit(
        fit_function,
        generations[valid],
        curve[valid],
    )

    a = popt[0]
    b = 0

    errors = np.sqrt(
        np.diag(pcov)
    )

    a_error = errors[0]
    b_error = 0

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        generations,
        curve,
        linewidth=2,
        color="k",
        label="Average",
    )

    ax.plot(
        generations,
        fit_function(
            generations,
            a,
        ),
        linestyle="dashed",
        label=(
            f"Fit: a = {a:.3f} +/- {a_error:.3f}"
        ),
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean heterozygosity")
    ax.set_title(title)

    ax.set_xlim(
        generations[0],
        generations[-1],
    )

    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_file = directory / f"{filename}.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    if repeat_number != -1:

        output_file = (
            directory
            / "../../../../"
            / f"{filename}_Landscape_{repeat_number}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()

    print(
        f"  Plot written to:\n"
        f"    {output_file}"
    )

    return a, a_error, b, b_error


# ============================================================================
# PROCESS ONE MC RUN
# ============================================================================

def process_run(
    run_directory,
    no_heatmaps,
):
    """
    Process one Monte-Carlo run.

    Returns:
        heterozygosity_matrix
        n_matrix
        heterozygosity_curve       # gamma diversity
        population_curve
        generations
        patches
        alpha_heterozygosity       # patch-indexed time series
        beta_diversity              # generation x patch x patch
    """

    print(
        f"\n{'=' * 48}\n"
        f"Processing: {run_directory.name}\n"
        f"{'=' * 48}"
    )

    ind_files = find_ind_files(
        run_directory
    )

    generations = [
        generation
        for generation, _ in ind_files
    ]

    print(
        f"Generations: "
        f"{generations[0]} -> {generations[-1]}"
    )

    # ------------------------------------------------------------------------
    # Detect loci
    # ------------------------------------------------------------------------

    first_file = ind_files[0][1]

    columns = pd.read_csv(
        first_file,
        nrows=0,
    ).columns

    locus_numbers = find_loci(
        columns
    )

    print(
        f"  Found {len(locus_numbers)} loci: "
        f"{locus_numbers}"
    )

    locus_columns = [
        f"L{locus}A{allele}"
        for locus in locus_numbers
        for allele in (0, 1)
    ]

    usecols = [
        "PatchID"
    ] + locus_columns

    # ------------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------------

    patch_data = set()

    heterozygosity_results = {}
    n_results = {}

    # Allele counts:
    #
    # allele_counts[generation][patch][locus]
    #
    # contains the total number of A1 alleles.
    allele_counts = {}

    # ------------------------------------------------------------------------
    # Process every generation once
    # ------------------------------------------------------------------------

    for generation, file in ind_files:

        print(
            f"  Calculating generation {generation}"
        )

        h_sum = {}
        h_count = {}

        generation_allele_counts = {}

        for chunk in pd.read_csv(
            file,
            usecols=usecols,
            dtype={
                "PatchID": np.int32,
                **{
                    column: np.int8
                    for column in locus_columns
                },
            },
            chunksize=100_000,
        ):

            # ---------------------------------------------------------------
            # Individual heterozygosity
            # ---------------------------------------------------------------

            individual_h = np.zeros(
                len(chunk),
                dtype=np.float32,
            )

            for locus in locus_numbers:

                heterozygous = (
                    (chunk[f"L{locus}A0"] == 1)
                    & (chunk[f"L{locus}A1"] == 1)
                )

                individual_h += (
                    heterozygous.to_numpy(
                        dtype=np.float32
                    )
                )

            individual_h /= len(
                locus_numbers
            )

            # ---------------------------------------------------------------
            # Group individuals by patch
            # ---------------------------------------------------------------

            temp = pd.DataFrame({
                "PatchID": chunk["PatchID"].to_numpy(),
                "H": individual_h,
            })

            grouped = temp.groupby(
                "PatchID"
            )["H"].agg(
                ["sum", "count"]
            )

            for patch, values in grouped.iterrows():

                h_sum[patch] = (
                    h_sum.get(patch, 0.0)
                    + values["sum"]
                )

                h_count[patch] = (
                    h_count.get(patch, 0)
                    + values["count"]
                )

            # ---------------------------------------------------------------
            # Allele counts for beta diversity
            # ---------------------------------------------------------------

            for patch, patch_chunk in chunk.groupby(
                "PatchID"
            ):

                if patch not in generation_allele_counts:

                    generation_allele_counts[patch] = {
                        locus: 0.0
                        for locus in locus_numbers
                    }

                for locus in locus_numbers:

                    generation_allele_counts[
                        patch
                    ][locus] += patch_chunk[
                        f"L{locus}A1"
                    ].sum()

        # --------------------------------------------------------------------
        # Store generation results
        # --------------------------------------------------------------------

        heterozygosity_results[generation] = {
            patch: h_sum[patch] / h_count[patch]
            for patch in h_sum
        }

        n_results[generation] = h_count

        allele_counts[generation] = (
            generation_allele_counts
        )

        patch_data.update(
            h_sum.keys()
        )

    patches = sorted(
        patch_data
    )

    print(
        f"  Found {len(patches)} patches."
    )

    # =========================================================================
    # BUILD RESULT ARRAYS
    # =========================================================================

    heterozygosity_array = np.full(
        (
            len(generations),
            len(patches),
        ),
        np.nan,
        dtype=np.float32,
    )

    n_array = np.zeros(
        (
            len(generations),
            len(patches),
        ),
        dtype=np.float32,
    )

    patch_index = {
        patch: i
        for i, patch in enumerate(patches)
    }

    for row, generation in enumerate(
        generations
    ):

        for patch, value in heterozygosity_results[
            generation
        ].items():

            heterozygosity_array[
                row,
                patch_index[patch],
            ] = value

        for patch, value in n_results[
            generation
        ].items():

            n_array[
                row,
                patch_index[patch],
            ] = value

    heterozygosity_matrix = pd.DataFrame(
        heterozygosity_array,
        index=generations,
        columns=patches,
    )

    n_matrix = pd.DataFrame(
        n_array,
        index=generations,
        columns=patches,
    )

    # =========================================================================
    # ALPHA DIVERSITY
    # =========================================================================

    # List indexed by actual PatchID.
    #
    # alpha_heterozygosity[patch_id] =
    #     [H(t0), H(t1), H(t2), ...]
    #
    # Empty/unrepresented patch IDs are left as [].

    max_patch_id = max(patches)

    alpha_heterozygosity = [
        []
        for _ in range(max_patch_id + 1)
    ]

    for patch in patches:

        alpha_heterozygosity[
            patch
        ] = heterozygosity_matrix[
            patch
        ].tolist()

    # =========================================================================
    # GAMMA DIVERSITY
    # =========================================================================

    # Existing landscape-wide heterozygosity.

    heterozygosity_curve = (
        heterozygosity_matrix
        .mean(
            axis=1,
            skipna=True,
        )
        .to_numpy()
    )

    population_curve = (
        n_matrix
        .mean(axis=1)
        .to_numpy()
    )

    # =========================================================================
    # BETA DIVERSITY
    # =========================================================================

    beta_diversity = calculate_beta_diversity(
        allele_counts,
        n_results,
        generations,
        patches,
        locus_numbers,
    )

    print(
        f"  Beta-diversity shape: "
        f"{beta_diversity.shape}"
    )

    # =========================================================================
    # HEATMAPS
    # =========================================================================

    if not no_heatmaps:

        plot_heatmap(
            heterozygosity_matrix,
            generations,
            patches,
            run_directory,
            "Population heterozygosity",
            "heterozygosity",
            "Mean heterozygosity",
            vmin=0,
            vmax=1,
            decimals=2,
        )

        plot_heatmap(
            n_matrix,
            generations,
            patches,
            run_directory,
            "Population Size",
            "N",
            "Mean number of individuals",
            vmin=0,
            vmax=None,
            decimals=1,
        )

    # =========================================================================
    # SAVE INDIVIDUAL MC-RUN DATA
    # =========================================================================

    heterozygosity_matrix.to_csv(
        run_directory / "heterozygosity.csv"
    )

    n_matrix.to_csv(
        run_directory / "N.csv"
    )

    # Alpha diversity:
    #
    # rows    = Patch ID
    # columns = Generation
    #
    # This is convenient for inspection while the returned object remains
    # a list indexed directly by PatchID.

    alpha_dataframe = pd.DataFrame(
        alpha_heterozygosity,
    )

    alpha_dataframe.index.name = "PatchID"
    alpha_dataframe.columns = generations

    alpha_dataframe.to_csv(
        run_directory / "alpha_heterozygosity.csv"
    )

    # Beta diversity:
    #
    # Save the complete 3-D array:
    #
    #     generation x patch x patch
    #
    np.save(
        run_directory / "beta_diversity.npy",
        beta_diversity,
    )

    print(
        f"  Data written to {run_directory}"
    )

    return (
        heterozygosity_matrix,
        n_matrix,
        heterozygosity_curve,
        population_curve,
        generations,
        patches,
        alpha_heterozygosity,
        beta_diversity,
    )


# ============================================================================
# SAVE COLLATED DATA
# ============================================================================

def save_data(
    output_dir,
    generations,
    patches,
    heterozygosity_curves,
    average_heterozygosity_curve,
    population_curves,
    average_population_curve,
    average_heterozygosity_matrix,
    average_n_matrix,
    alpha_heterozygosity,
    beta_diversity,
):
    """Save all data required for later analysis."""

    output_file = (
        output_dir / "Data.npz"
    )

    np.savez_compressed(
        output_file,

        generations=np.asarray(
            generations
        ),

        patches=np.asarray(
            patches
        ),

        heterozygosity_curves=np.asarray(
            heterozygosity_curves
        ),

        average_heterozygosity_curve=np.asarray(
            average_heterozygosity_curve
        ),

        population_curves=np.asarray(
            population_curves
        ),

        average_population_curve=np.asarray(
            average_population_curve
        ),

        average_heterozygosity_matrix=np.asarray(
            average_heterozygosity_matrix
        ),

        average_n_matrix=np.asarray(
            average_n_matrix
        ),

        # ---------------------------------------------------------------
        # NEW:
        #
        # alpha_heterozygosity:
        #     MC x PatchID x Generation
        #
        # beta_diversity:
        #     MC x Generation x Patch x Patch
        # ---------------------------------------------------------------

        alpha_heterozygosity=np.asarray(
            alpha_heterozygosity,
            dtype=object,
        ),

        beta_diversity=np.asarray(
            beta_diversity
        ),
    )

    print(
        f"\nCollated data written to:\n"
        f"  {output_file}"
    )


# ============================================================================
# ANALYSE EXISTING DATA.NPZ
# ============================================================================

def analyse_npz(
    output_dir,
    no_heatmaps,
    no_plots,
    repeat_number=-1,
):
    """
    Analyse existing Data.npz.

    The Data.npz must have been generated by this version of the script,
    because alpha_heterozygosity and beta_diversity are new outputs.
    """

    data_file = (
        output_dir / "Data.npz"
    )

    print(
        f"\nFound existing {data_file}"
    )

    print(
        "Using Data.npz; "
        "ind*.csv files will NOT be read."
    )

    with np.load(
        data_file,
        allow_pickle=True,
    ) as data:

        generations = data[
            "generations"
        ]

        patches = data[
            "patches"
        ].tolist()

        heterozygosity_curves = data[
            "heterozygosity_curves"
        ]

        average_heterozygosity_curve = data[
            "average_heterozygosity_curve"
        ]

        population_curves = data[
            "population_curves"
        ]

        average_population_curve = data[
            "average_population_curve"
        ]

        average_heterozygosity_matrix = (
            pd.DataFrame(
                data[
                    "average_heterozygosity_matrix"
                ],
                index=generations,
                columns=patches,
            )
        )

        average_n_matrix = pd.DataFrame(
            data[
                "average_n_matrix"
            ],
            index=generations,
            columns=patches,
        )

        # ---------------------------------------------------------------
        # New data
        # ---------------------------------------------------------------

        if "alpha_heterozygosity" not in data:
            raise ValueError(
                "This Data.npz does not contain "
                "alpha_heterozygosity. Delete Data.npz "
                "and rerun the analysis."
            )

        if "beta_diversity" not in data:
            raise ValueError(
                "This Data.npz does not contain "
                "beta_diversity. Delete Data.npz "
                "and rerun the analysis."
            )

        alpha_heterozygosity = data[
            "alpha_heterozygosity"
        ]

        beta_diversity = data[
            "beta_diversity"
        ]

    number_runs = len(
        heterozygosity_curves
    )

    print(
        f"  Loaded {number_runs} "
        f"heterozygosity curves."
    )

    print(
        f"  Alpha diversity shape: "
        f"{alpha_heterozygosity.shape}"
    )

    print(
        f"  Beta diversity shape: "
        f"{beta_diversity.shape}"
    )

    # =========================================================================
    # HEATMAPS
    # =========================================================================

    if not no_heatmaps:

        plot_heatmap(
            average_heterozygosity_matrix,
            generations,
            patches,
            output_dir,
            f"AVERAGED population heterozygosity\n"
            f"{number_runs} Monte-Carlo runs",
            "heterozygosity_AVERAGE",
            "Mean heterozygosity",
            vmin=0,
            vmax=1,
            decimals=2,
        )

        plot_heatmap(
            average_n_matrix,
            generations,
            patches,
            output_dir,
            f"AVERAGED number of individuals\n"
            f"{number_runs} Monte-Carlo runs",
            "N_AVERAGE",
            "Mean number of individuals",
            vmin=0,
            vmax=None,
            decimals=1,
        )

    # =========================================================================
    # FIT
    # =========================================================================

    a, a_error, b, b_error = (
        fit_heterozygosity_curve(
            generations,
            average_heterozygosity_curve,
            output_dir,
            f"Landscape-wide average heterozygosity\n"
            f"{number_runs} Monte-Carlo runs",
            "heterozygosity_AVERAGE_landscape_average",
            repeat_number=repeat_number,
        )
    )

    # =========================================================================
    # PLOTS
    # =========================================================================

    if not no_plots:

        plot_curves(
            generations,
            heterozygosity_curves,
            average_heterozygosity_curve,
            output_dir,
            "heterozygosity_ALL_curves",
            "Mean heterozygosity",
            f"Landscape-wide heterozygosity\n"
            f"{number_runs} Monte-Carlo runs",
            log=True,
            repeat_number=repeat_number,
        )

        plot_curves(
            generations,
            population_curves,
            average_population_curve,
            output_dir,
            "N_landscape_average",
            "Mean number of individuals",
            f"Landscape-wide population size\n"
            f"{number_runs} Monte-Carlo runs",
            repeat_number=repeat_number,
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(
        "\n" + "=" * 48
    )

    print(
        "Data.npz analysis complete"
    )

    print(
        "=" * 48
    )

    print(
        f"a = {a:.6f}"
    )

    print(
        f"Error on a = {a_error:.6f}"
    )

    print(
        f"\nAlpha diversity shape: "
        f"{alpha_heterozygosity.shape}"
    )

    print(
        f"Beta diversity shape: "
        f"{beta_diversity.shape}"
    )

    return (
        a,
        average_heterozygosity_curve,
        average_population_curve,
        alpha_heterozygosity,
        beta_diversity,
    )


# ============================================================================
# MAIN
# ============================================================================

def main(
    d,
    no_heatmaps=False,
    no_plots=False,
    no_save=False,
    repeat_number=-1,
):

    output_dir = Path(d)

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {output_dir}"
        )

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {output_dir}"
        )

    # ------------------------------------------------------------------------
    # If Data.npz already exists, analyse it directly.
    # ------------------------------------------------------------------------

    data_file = (
        output_dir / "Data.npz"
    )

    if data_file.exists():

        return analyse_npz(
            output_dir,
            no_heatmaps,
            no_plots,
            repeat_number=repeat_number,
        )

    # ------------------------------------------------------------------------
    # Find MC runs
    # ------------------------------------------------------------------------

    run_directories = find_runs(
        output_dir
    )

    print(
        f"\nFound {len(run_directories)} "
        f"Monte-Carlo runs."
    )

    for mc, directory in run_directories:

        print(
            f"  MC {mc}: {directory.name}"
        )

    # =========================================================================
    # STORAGE FOR MC RESULTS
    # =========================================================================

    heterozygosity_matrices = []
    n_matrices = []

    heterozygosity_curves = []
    population_curves = []

    # ------------------------------------------------------------------------
    # NEW:
    #
    # One alpha time-series list per MC run.
    #
    # One beta array per MC run:
    #
    #     generation x patch x patch
    # ------------------------------------------------------------------------

    alpha_heterozygosity_all = []
    beta_diversity_all = []

    common_generations = None
    common_patches = None

    # =========================================================================
    # PROCESS EACH MC RUN
    # =========================================================================

    for mc, run_directory in run_directories:

        (
            h_matrix,
            n_matrix,
            h_curve,
            n_curve,
            generations,
            patches,
            alpha_heterozygosity,
            beta_diversity,
        ) = process_run(
            run_directory,
            no_heatmaps,
        )

        # --------------------------------------------------------------------
        # Check dimensions
        # --------------------------------------------------------------------

        if common_generations is None:

            common_generations = generations
            common_patches = patches

        else:

            if generations != common_generations:

                raise ValueError(
                    "Generation mismatch between "
                    f"MC runs.\n"
                    f"Expected: {common_generations}\n"
                    f"Found: {generations}\n"
                    f"Problem run: {run_directory}"
                )

            if patches != common_patches:

                raise ValueError(
                    "Patch mismatch between "
                    f"MC runs.\n"
                    f"Expected: {common_patches}\n"
                    f"Found: {patches}\n"
                    f"Problem run: {run_directory}"
                )

        # --------------------------------------------------------------------
        # Store results
        # --------------------------------------------------------------------

        heterozygosity_matrices.append(
            h_matrix
        )

        n_matrices.append(
            n_matrix
        )

        heterozygosity_curves.append(
            h_curve
        )

        population_curves.append(
            n_curve
        )

        alpha_heterozygosity_all.append(
            alpha_heterozygosity
        )

        beta_diversity_all.append(
            beta_diversity
        )

    number_runs = len(
        heterozygosity_matrices
    )

    # =========================================================================
    # COMBINE MC RUNS
    # =========================================================================

    print(
        "\n" + "=" * 48 +
        "\nCombining Monte-Carlo runs...\n" +
        "=" * 48
    )

    # ------------------------------------------------------------------------
    # Average heterozygosity matrix
    # ------------------------------------------------------------------------

    h_stack = np.stack(
        [
            matrix.to_numpy(
                dtype=float
            )
            for matrix in heterozygosity_matrices
        ]
    )

    average_heterozygosity_matrix = (
        pd.DataFrame(
            np.nanmean(
                h_stack,
                axis=0,
            ),
            index=common_generations,
            columns=common_patches,
        )
    )

    # ------------------------------------------------------------------------
    # Average N matrix
    # ------------------------------------------------------------------------

    n_stack = np.stack(
        [
            matrix.to_numpy(
                dtype=float
            )
            for matrix in n_matrices
        ]
    )

    average_n_matrix = pd.DataFrame(
        np.mean(
            n_stack,
            axis=0,
        ),
        index=common_generations,
        columns=common_patches,
    )

    # ------------------------------------------------------------------------
    # Average landscape curves
    # ------------------------------------------------------------------------

    heterozygosity_curves = np.asarray(
        heterozygosity_curves,
        dtype=float,
    )

    population_curves = np.asarray(
        population_curves,
        dtype=float,
    )

    average_heterozygosity_curve = (
        np.mean(
            heterozygosity_curves,
            axis=0,
        )
    )

    average_population_curve = (
        np.mean(
            population_curves,
            axis=0,
        )
    )

    # ------------------------------------------------------------------------
    # DO NOT AVERAGE ALPHA OR BETA
    #
    # Keep every MC replicate separately.
    # ------------------------------------------------------------------------

    beta_diversity_all = np.stack(
        beta_diversity_all
    )

    print(
        f"\nBeta-diversity data retained as:"
        f"\n  {beta_diversity_all.shape}"
    )

    # =========================================================================
    # SAVE DATA
    # =========================================================================

    if not no_save:

        save_data(
            output_dir,
            common_generations,
            common_patches,
            heterozygosity_curves,
            average_heterozygosity_curve,
            population_curves,
            average_population_curve,
            average_heterozygosity_matrix,
            average_n_matrix,
            alpha_heterozygosity_all,
            beta_diversity_all,
        )

        # --------------------------------------------------------------------
        # Existing CSV outputs
        # --------------------------------------------------------------------

        average_heterozygosity_matrix.to_csv(
            output_dir
            / "heterozygosity_AVERAGE.csv"
        )

        average_n_matrix.to_csv(
            output_dir
            / "N_AVERAGE.csv"
        )

        print(
            "\nAverage heterozygosity data written to:"
            f"\n  {output_dir / 'heterozygosity_AVERAGE.csv'}"
        )

        print(
            "\nAverage population-size data written to:"
            f"\n  {output_dir / 'N_AVERAGE.csv'}"
        )

        # --------------------------------------------------------------------
        # Save complete beta data at the top level too
        # --------------------------------------------------------------------

        np.save(
            output_dir
            / "beta_diversity_ALL.npy",
            beta_diversity_all,
        )

        print(
            "\nBeta-diversity data written to:"
            f"\n  {output_dir / 'beta_diversity_ALL.npy'}"
        )

    # =========================================================================
    # PLOTS
    # =========================================================================

    if not no_heatmaps:

        plot_heatmap(
            average_heterozygosity_matrix,
            common_generations,
            common_patches,
            output_dir,
            f"AVERAGED population heterozygosity\n"
            f"{number_runs} Monte-Carlo runs",
            "heterozygosity_AVERAGE",
            "Mean heterozygosity",
            vmin=0,
            vmax=1,
            decimals=2,
        )

        plot_heatmap(
            average_n_matrix,
            common_generations,
            common_patches,
            output_dir,
            f"AVERAGED number of individuals\n"
            f"{number_runs} Monte-Carlo runs",
            "N_AVERAGE",
            "Mean number of individuals",
            vmin=0,
            vmax=None,
            decimals=1,
        )

    # ------------------------------------------------------------------------
    # Averaged heterozygosity + fit
    # ------------------------------------------------------------------------

    a, a_error, b, b_error = (
        fit_heterozygosity_curve(
            common_generations,
            average_heterozygosity_curve,
            output_dir,
            f"Landscape-wide average heterozygosity\n"
            f"{number_runs} Monte-Carlo runs",
            "heterozygosity_AVERAGE_landscape_average",
            repeat_number=repeat_number,
        )
    )

    # ------------------------------------------------------------------------
    # All heterozygosity curves
    # ------------------------------------------------------------------------

    if not no_plots:

        plot_curves(
            common_generations,
            heterozygosity_curves,
            average_heterozygosity_curve,
            output_dir,
            "heterozygosity_ALL_curves",
            "Mean heterozygosity",
            f"Landscape-wide heterozygosity\n"
            f"{number_runs} Monte-Carlo runs",
            log=True,
            repeat_number=repeat_number,
        )

        plot_curves(
            common_generations,
            population_curves,
            average_population_curve,
            output_dir,
            "N_landscape_average",
            "Mean number of individuals",
            f"Landscape-wide population size\n"
            f"{number_runs} Monte-Carlo runs",
            repeat_number=repeat_number,
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print(
        "\n" + "=" * 48
    )

    print(
        "Finished."
    )

    print(
        "=" * 48
    )

    print(
        f"Monte-Carlo runs processed: "
        f"{number_runs}"
    )

    print(
        f"Generations: "
        f"{common_generations[0]} -> "
        f"{common_generations[-1]}"
    )

    print(
        f"Number of patches: "
        f"{len(common_patches)}"
    )

    print(
        "\nData:"
    )

    print(
        f"  gamma heterozygosity curves: "
        f"{heterozygosity_curves.shape}"
    )

    print(
        f"  alpha heterozygosity: "
        f"{len(alpha_heterozygosity_all)} MC runs"
    )

    print(
        f"  beta diversity: "
        f"{beta_diversity_all.shape}"
    )

    print(
        "\nFit:"
    )

    print(
        f"  a = {a:.6f} +/- {a_error:.6f}"
    )

    print(
        f"  b = {b:.6f} +/- {b_error:.6f}"
    )

    return (
        a,
        average_heterozygosity_curve,
        average_population_curve,
        alpha_heterozygosity_all,
        beta_diversity_all,
    )


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    args = parser.parse_args()

    main(
        d=args.d,
        no_heatmaps=args.no_heatmaps,
        no_plots=args.no_plots,
        no_save=args.no_save,
        repeat_number=-1,
    )
