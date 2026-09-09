import argparse
import gc
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
    description="Calculate population heterozygosity and population size."
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


def calculate_heterozygosity(df):
    """Calculate individual heterozygosity across the two loci."""

    locus_0 = (df["L0A0"] == 1) & (df["L0A1"] == 1)
    locus_1 = (df["L1A0"] == 1) & (df["L1A1"] == 1)

    df["Heterozygosity"] = (
        locus_0.astype(float) +
        locus_1.astype(float)
    ) / 2.0

    return df


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

    print(f"  Plot written to:\n    {output_file}")


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
        output_file = directory / "../../../../" / f"{filename}_Landscape_{repeat_number}.png"
        plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )

    plt.close()

    print(f"  Plot written to:\n    {output_file}")


def fit_heterozygosity_curve(
    generations,
    curve,
    directory,
    title,
    filename,
    repeat_number=-1
):
    """Fit b * exp(-x / (2a)) to the averaged curve."""

    generations = np.asarray(
        generations,
        dtype=float,
    )

    curve = np.asarray(
        curve,
        dtype=float,
    )

    def fit_function(x, a, b):
        return b * np.exp(-x / (2 * a))

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

    a, b = popt

    errors = np.sqrt(
        np.diag(pcov)
    )

    a_error = errors[0]
    b_error = errors[1]

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
            b,
        ),
        linestyle="dashed",
        label=(
            f"Fit: a = {a:.3f} ± {a_error:.3f}"
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
        output_file = directory / "../../../../" / f"{filename}_Landscape_{repeat_number}.png"
        plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )
    

    plt.close()

    print(f"  Plot written to:\n    {output_file}")

    return a, a_error, b, b_error


# ============================================================================
# PROCESS ONE MC RUN
# ============================================================================

def process_run(run_directory, no_heatmaps):

    print(
        f"\n{'=' * 48}\n"
        f"Processing: {run_directory.name}\n"
        f"{'=' * 48}"
    )

    ind_files = find_ind_files(run_directory)

    generations = [
        generation
        for generation, _ in ind_files
    ]

    print(
        f"Generations: "
        f"{generations[0]} -> {generations[-1]}"
    )

    # We build the patch list as we process the files.
    # Results are temporarily stored by patch ID.
    # We build the patch list as we process the files.
    patch_data = set()

    heterozygosity_results = {}
    n_results = {}

    # Process each generation ONCE
    for generation, file in ind_files:

        print(f"  Calculating generation {generation}")

        h_sum = {}
        h_count = {}

        for chunk in pd.read_csv(
            file,
            usecols=[
                "PatchID",
                "L0A0",
                "L0A1",
                "L1A0",
                "L1A1",
            ],
            dtype={
                "PatchID": np.int32,
                "L0A0": np.int8,
                "L0A1": np.int8,
                "L1A0": np.int8,
                "L1A1": np.int8,
            },
            chunksize=100_000,
        ):

            locus_0 = (
                (chunk["L0A0"] == 1)
                & (chunk["L0A1"] == 1)
            )

            locus_1 = (
                (chunk["L1A0"] == 1)
                & (chunk["L1A1"] == 1)
            )

            individual_h = (
                locus_0.astype(np.float32)
                + locus_1.astype(np.float32)
            ) / 2.0

            temp = pd.DataFrame({
                "PatchID": chunk["PatchID"].to_numpy(),
                "H": individual_h.to_numpy(),
            })

            grouped = temp.groupby("PatchID")["H"].agg(
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

        heterozygosity_results[generation] = {
            patch: h_sum[patch] / h_count[patch]
            for patch in h_sum
        }

        n_results[generation] = h_count

        patch_data.update(h_sum.keys())

    patches = sorted(patch_data)

    print(f"  Found {len(patches)} patches.")

    # Build result arrays
    heterozygosity_array = np.full(
        (len(generations), len(patches)),
        np.nan,
        dtype=np.float32,
    )

    n_array = np.zeros(
        (len(generations), len(patches)),
        dtype=np.float32,
    )

    patch_index = {
        patch: i
        for i, patch in enumerate(patches)
    }

    for row, generation in enumerate(generations):

        for patch, value in heterozygosity_results[
            generation
        ].items():

            heterozygosity_array[
                row,
                patch_index[patch]
            ] = value

        for patch, value in n_results[
            generation
        ].items():

            n_array[
                row,
                patch_index[patch]
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

    # Landscape curves
    heterozygosity_curve = (
        heterozygosity_matrix
        .mean(axis=1, skipna=True)
        .to_numpy()
    )

    population_curve = (
        n_matrix
        .mean(axis=1)
        .to_numpy()
    )

    # Heatmaps
    if not no_heatmaps:

        plot_heatmap(
            heterozygosity_matrix,
            "Heterozygosity",
            run_directory / "heterozygosity.png",
        )

        plot_heatmap(
            n_matrix,
            "Population Size",
            run_directory / "N.png",
        )

    # Save matrices
    heterozygosity_matrix.to_csv(
        run_directory / "heterozygosity.csv"
    )

    n_matrix.to_csv(
        run_directory / "N.csv"
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
):
    """Save all data required for later analysis."""

    output_file = output_dir / "Data.npz"

    np.savez_compressed(
        output_file,
        generations=np.asarray(generations),
        patches=np.asarray(patches),
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
    repeat_number=-1
):
    """Analyse Data.npz without reading ind*.csv files."""

    data_file = output_dir / "Data.npz"

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

        average_heterozygosity_matrix = pd.DataFrame(
            data["average_heterozygosity_matrix"],
            index=generations,
            columns=patches,
        )

        average_n_matrix = pd.DataFrame(
            data["average_n_matrix"],
            index=generations,
            columns=patches,
        )

    number_runs = len(
        heterozygosity_curves
    )

    print(
        f"  Loaded {number_runs} heterozygosity curves."
    )

    print(heterozygosity_curves)

    # ------------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Curves
    # ------------------------------------------------------------------------

    a, a_error, b, b_error = fit_heterozygosity_curve(
        generations,
        average_heterozygosity_curve,
        output_dir,
        f"Landscape-wide average heterozygosity\n"
        f"{number_runs} Monte-Carlo runs",
        "heterozygosity_AVERAGE_landscape_average",
        repeat_number=repeat_number
    )

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
        repeat_number=repeat_number
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
        repeat_number=repeat_number
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 48)
    print("Data.npz analysis complete")
    print("=" * 48)

    print(f"a = {a:.6f}")
    print(f"Error on a = {a_error:.6f}")
    print(f"b = {b:.6f}")
    print(f"Error on b = {b_error:.6f}")

    print(
        f"\nLoaded heterozygosity curves: "
        f"{len(heterozygosity_curves)}"
    )

    print(
        f"Curve length: "
        f"{len(average_heterozygosity_curve)}"
    )

    return a


# ============================================================================
# MAIN
# ============================================================================

def main(d, no_heatmaps=False,repeat_number=-1):

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

    data_file = output_dir / "Data.npz"

    if data_file.exists():
        return analyse_npz(
            output_dir,
            no_heatmaps,
            repeat_number=repeat_number
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

    # ------------------------------------------------------------------------
    # Storage for MC results
    # ------------------------------------------------------------------------

    heterozygosity_matrices = []
    n_matrices = []

    heterozygosity_curves = []
    population_curves = []

    common_generations = None
    common_patches = None

    # ------------------------------------------------------------------------
    # Process each MC run
    # ------------------------------------------------------------------------

    for mc, run_directory in run_directories:

        (
            h_matrix,
            n_matrix,
            h_curve,
            n_curve,
            generations,
            patches,
        ) = process_run(
            run_directory,
            no_heatmaps,
        )

        # Check dimensions

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

        # Store results

        heterozygosity_matrices.append(
            h_matrix
        )

        n_matrices.append(
            n_matrix
        )

        # IMPORTANT:
        # These are now actually populated.

        heterozygosity_curves.append(
            h_curve
        )

        population_curves.append(
            n_curve
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
    #
    # NaN = empty patch.
    # Empty patches therefore do not contribute.
    # ------------------------------------------------------------------------

    h_stack = np.stack(
        [
            matrix.to_numpy(dtype=float)
            for matrix in heterozygosity_matrices
        ]
    )

    average_heterozygosity_matrix = pd.DataFrame(
        np.nanmean(h_stack, axis=0),
        index=common_generations,
        columns=common_patches,
    )

    # ------------------------------------------------------------------------
    # Average N matrix
    #
    # Empty patches are already zero.
    # ------------------------------------------------------------------------

    n_stack = np.stack(
        [
            matrix.to_numpy(dtype=float)
            for matrix in n_matrices
        ]
    )

    average_n_matrix = pd.DataFrame(
        np.mean(n_stack, axis=0),
        index=common_generations,
        columns=common_patches,
    )

    # ------------------------------------------------------------------------
    # Average landscape curves
    #
    # IMPORTANT:
    # These are averages of the MC curves themselves.
    # ------------------------------------------------------------------------

    heterozygosity_curves = np.asarray(
        heterozygosity_curves,
        dtype=float,
    )

    population_curves = np.asarray(
        population_curves,
        dtype=float,
    )

    average_heterozygosity_curve = np.mean(
        heterozygosity_curves,
        axis=0,
    )

    average_population_curve = np.mean(
        population_curves,
        axis=0,
    )

    # =========================================================================
    # SAVE DATA
    # =========================================================================

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
    )

    # ------------------------------------------------------------------------
    # CSV outputs
    # ------------------------------------------------------------------------

    average_heterozygosity_matrix.to_csv(
        output_dir / "heterozygosity_AVERAGE.csv"
    )

    average_n_matrix.to_csv(
        output_dir / "N_AVERAGE.csv"
    )

    print(
        "\nAverage heterozygosity data written to:"
        f"\n  {output_dir / 'heterozygosity_AVERAGE.csv'}"
    )

    print(
        "\nAverage population-size data written to:"
        f"\n  {output_dir / 'N_AVERAGE.csv'}"
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

    a, a_error, b, b_error = fit_heterozygosity_curve(
        common_generations,
        average_heterozygosity_curve,
        output_dir,
        f"Landscape-wide average heterozygosity\n"
        f"{number_runs} Monte-Carlo runs",
        "heterozygosity_AVERAGE_landscape_average",
        repeat_number=repeat_number
    )

    # ------------------------------------------------------------------------
    # All heterozygosity curves
    # ------------------------------------------------------------------------

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
        repeat_number=repeat_number
    )

    # ------------------------------------------------------------------------
    # Population curves
    # ------------------------------------------------------------------------

    plot_curves(
        common_generations,
        population_curves,
        average_population_curve,
        output_dir,
        "N_landscape_average",
        "Mean number of individuals",
        f"Landscape-wide population size\n"
        f"{number_runs} Monte-Carlo runs",
        repeat_number=repeat_number
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 48)
    print("Finished.")
    print("=" * 48)

    print(
        f"Monte-Carlo runs processed: {number_runs}"
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

    print("\nData.npz contains:")
    print(
        f"  heterozygosity_curves: "
        f"{heterozygosity_curves.shape}"
    )
    print(
        f"  average_heterozygosity_curve: "
        f"{average_heterozygosity_curve.shape}"
    )
    print(
        f"  population_curves: "
        f"{population_curves.shape}"
    )
    print(
        f"  average_population_curve: "
        f"{average_population_curve.shape}"
    )

    print("\nFit:")
    print(f"  a = {a:.6f} ± {a_error:.6f}")
    print(f"  b = {b:.6f} ± {b_error:.6f}")

    return a


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    args = parser.parse_args()

    main(
        d=args.d,
        no_heatmaps=args.no_heatmaps,
        repeat_number=-1
    )
