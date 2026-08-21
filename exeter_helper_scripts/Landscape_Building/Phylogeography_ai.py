import pandas as pd
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gc


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Track spatial ancestry across Monte-Carlo runs."
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing the Monte-Carlo run directories"
)

parser.add_argument(
    "-g",
    type=str,
    required=True,
    choices=[
        "all",
        "L0A0A0",
        "L0A0A1",
        "L0A1A1",
        "L1A0A0",
        "L1A0A1",
        "L1A1A1"
    ],
    help="The genotype in question, or 'all' to use every individual"
)

parser.add_argument(
    "-PID",
    type=str,
    required=True,
    help="The patch ID"
)

parser.add_argument(
    "--counts",
    action="store_true",
    help="Show raw ancestry counts as text inside the heatmap cells"
)

parser.add_argument(
    "--unique",
    action="store_true",
    help="Count each individual only once within each generation"
)

parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Do not create heatmaps"
)


args = parser.parse_args()


# ============================================================
# CHECK DIRECTORY
# ============================================================

output_dir = Path(
    args.d
)

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

        run_number = int(
            match.group(1)
        )

        batch_number = int(
            match.group(2)
        )

        mc_number = int(
            match.group(3)
        )

        species_number = int(
            match.group(4)
        )


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
    f"Found {len(run_directories)} "
    f"Monte-Carlo runs."
)

for mc_number, directory in run_directories:

    print(
        f"  MC {mc_number}: "
        f"{directory.name}"
    )


# ============================================================
# FUNCTION TO EXTRACT LAST THREE ID FIELDS
# ============================================================

def get_last_three(
    id_string
):

    """
    Convert a CDMetaPOP ID to its final three components.

    Example:

        RD1_F3_m3f2_P3_Y8_UO52

    becomes:

        P3_Y8_UO52
    """

    if pd.isna(
        id_string
    ):

        return None


    parts = str(
        id_string
    ).split("_")


    if len(parts) <= 3:

        return str(
            id_string
        )


    return "_".join(
        parts[-3:]
    )


# ============================================================
# FUNCTION TO CHECK GENOTYPE
# ============================================================

def genotype_matches(
    row,
    genotype
):

    if genotype == "L0A0A0":

        return (
            row["L0A0"] == 2
            and row["L0A1"] == 0
        )


    elif genotype == "L0A0A1":

        return (
            row["L0A0"] == 1
            and row["L0A1"] == 1
        )


    elif genotype == "L0A1A1":

        return (
            row["L0A0"] == 0
            and row["L0A1"] == 2
        )


    elif genotype == "L1A0A0":

        return (
            row["L1A0"] == 2
            and row["L1A1"] == 0
        )


    elif genotype == "L1A0A1":

        return (
            row["L1A0"] == 1
            and row["L1A1"] == 1
        )


    elif genotype == "L1A1A1":

        return (
            row["L1A0"] == 0
            and row["L1A1"] == 2
        )


    else:

        raise ValueError(
            f"Unknown genotype: {genotype}"
        )


# ============================================================
# FIND IND FILES
# ============================================================

def find_ind_files(
    run_directory
):

    ind_files = []


    for file in run_directory.glob(
        "ind*.csv"
    ):

        match = re.fullmatch(
            r"ind(\d+)\.csv",
            file.name
        )


        if match:

            generation = int(
                match.group(1)
            )


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
# FIND ALL PATCHES IN A RUN
# ============================================================

def find_all_patches(
    ind_data,
    generations
):

    all_patches = set()


    for generation in generations:

        df = ind_data[
            generation
        ]


        for patch in df[
            "PatchID"
        ].unique():

            all_patches.add(
                patch
            )


    all_patches = sorted(
        all_patches
    )


    return all_patches


# ============================================================
# CHECK WHETHER PARENT ID IS VALID
# ============================================================

def valid_parent_id(
    parent_id
):

    if pd.isna(
        parent_id
    ):

        return False


    if str(
        parent_id
    ) == "":

        return False


    if str(
        parent_id
    ).lower() == "nan":

        return False


    return True


# ============================================================
# TRACE ONE INDIVIDUAL BACKWARDS
# ============================================================

def trace_individual(
    individual_id,
    start_generation,
    id_lookup,
    individual_cache
):

    individual_id = get_last_three(
        individual_id
    )


    cache_key = (
        individual_id,
        start_generation
    )


    # --------------------------------------------------------
    # CHECK CACHE
    # --------------------------------------------------------

    if cache_key in individual_cache:

        return individual_cache[
            cache_key
        ]


    history = []

    mother_id = None

    father_id = None


    current_generation = (
        start_generation
    )


    # --------------------------------------------------------
    # SEARCH BACKWARDS
    # --------------------------------------------------------

    while current_generation >= 0:

        generation_lookup = id_lookup.get(
            current_generation,
            {}
        )


        # ----------------------------------------------------
        # INDIVIDUAL HAS DISAPPEARED
        # ----------------------------------------------------

        if individual_id not in generation_lookup:

            break


        # ----------------------------------------------------
        # INDIVIDUAL EXISTS
        # ----------------------------------------------------

        row = generation_lookup[
            individual_id
        ]


        # ----------------------------------------------------
        # Record location
        # ----------------------------------------------------

        history.append(
            {
                "Year": current_generation,

                "ID": individual_id,

                "PatchID": row["PatchID"]
            }
        )


        # ----------------------------------------------------
        # Store parents
        # ----------------------------------------------------

        mother_id = get_last_three(
            row["MID"]
        )

        father_id = get_last_three(
            row["FID"]
        )


        # ----------------------------------------------------
        # Move backwards
        # ----------------------------------------------------

        current_generation -= 1


    # ========================================================
    # FIRST GENERATION WHERE INDIVIDUAL DOES NOT EXIST
    # ========================================================

    parent_start_generation = (
        current_generation
    )


    result = {

        "History": history,

        "Mother": mother_id,

        "Father": father_id,

        "ParentStartGeneration":
            parent_start_generation
    }


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    individual_cache[
        cache_key
    ] = result


    return result


# ============================================================
# TRACE COMPLETE ANCESTRY BRANCH
# ============================================================

def trace_ancestry_branch(
    individual_id,
    start_generation,
    id_lookup,
    individual_cache,
    ancestry_cache
):

    individual_id = get_last_three(
        individual_id
    )


    cache_key = (
        individual_id,
        start_generation
    )


    # --------------------------------------------------------
    # CHECK CACHE
    # --------------------------------------------------------

    if cache_key in ancestry_cache:

        return ancestry_cache[
            cache_key
        ]


    ancestry = []


    # ========================================================
    # TRACE THIS INDIVIDUAL
    # ========================================================

    result = trace_individual(
        individual_id,
        start_generation,
        id_lookup,
        individual_cache
    )


    # --------------------------------------------------------
    # Add individual's history
    # --------------------------------------------------------

    for entry in result[
        "History"
    ]:

        ancestry.append(
            entry.copy()
        )


    # ========================================================
    # DETERMINE WHERE TO START PARENTS
    # ========================================================

    parent_start_generation = (
        result[
            "ParentStartGeneration"
        ]
    )


    # --------------------------------------------------------
    # No earlier generation exists
    # --------------------------------------------------------

    if parent_start_generation < 0:

        ancestry_cache[
            cache_key
        ] = ancestry

        return ancestry


    # ========================================================
    # MOTHER
    # ========================================================

    mother_id = result[
        "Mother"
    ]


    if valid_parent_id(
        mother_id
    ):

        mother_id = get_last_three(
            mother_id
        )


        mother_branch = trace_ancestry_branch(
            mother_id,
            parent_start_generation,
            id_lookup,
            individual_cache,
            ancestry_cache
        )


        for entry in mother_branch:

            ancestry.append(
                entry.copy()
            )


    # ========================================================
    # FATHER
    # ========================================================

    father_id = result[
        "Father"
    ]


    if valid_parent_id(
        father_id
    ):

        father_id = get_last_three(
            father_id
        )


        father_branch = trace_ancestry_branch(
            father_id,
            parent_start_generation,
            id_lookup,
            individual_cache,
            ancestry_cache
        )


        for entry in father_branch:

            ancestry.append(
                entry.copy()
            )


    # ========================================================
    # CACHE COMPLETE BRANCH
    # ========================================================

    ancestry_cache[
        cache_key
    ] = ancestry


    return ancestry


# ============================================================
# PROCESS ONE MONTE-CARLO RUN
# ============================================================

def process_run(
    run_directory
):

    print()
    print(
        "================================================"
    )

    print(
        f"Processing: "
        f"{run_directory.name}"
    )

    print(
        "================================================"
    )


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


    final_generation = generations[-1]


    print(
        f"Final generation: "
        f"{final_generation}"
    )


    # ========================================================
    # READ IND FILES
    # ========================================================

    ind_data = {}


    for generation, file in ind_files:

        print(
            f"  Reading generation "
            f"{generation}"
        )


        df = pd.read_csv(
            file
        )


        ind_data[
            generation
        ] = df


    # ========================================================
    # CREATE ID LOOKUP
    # ========================================================

    print(
        "  Creating ID lookup..."
    )


    id_lookup = {}


    for generation in generations:

        id_lookup[
            generation
        ] = {}


        df = ind_data[
            generation
        ]


        for _, row in df.iterrows():

            individual_id = get_last_three(
                row["ID"]
            )


            id_lookup[
                generation
            ][
                individual_id
            ] = row


    # ========================================================
    # FIND PATCHES
    # ========================================================

    all_patches = find_all_patches(
        ind_data,
        generations
    )


    # ========================================================
    # FIND STARTING INDIVIDUALS
    # ========================================================

    print(
        "  Finding starting individuals..."
    )


    final_df = ind_data[
        final_generation
    ]


    starting_individuals = []


    for _, row in final_df.iterrows():

        # ----------------------------------------------------
        # Patch filter
        # ----------------------------------------------------

        if str(
            row["PatchID"]
        ) != args.PID:

            continue


        # ----------------------------------------------------
        # Genotype filter
        # ----------------------------------------------------

        if args.g == "all":

            genotype_match = True

        else:

            genotype_match = genotype_matches(
                row,
                args.g
            )


        if genotype_match:

            individual = {

                "ID": get_last_three(
                    row["ID"]
                ),

                "PatchID": row["PatchID"],

                "MID": get_last_three(
                    row["MID"]
                ),

                "FID": get_last_three(
                    row["FID"]
                )
            }


            starting_individuals.append(
                individual
            )


    print(
        f"  Starting individuals: "
        f"{len(starting_individuals)}"
    )


    if len(starting_individuals) == 0:

        raise ValueError(
            f"No starting individuals found "
            f"in PatchID {args.PID}"
        )


    # ========================================================
    # CACHES
    # ========================================================

    individual_cache = {}

    ancestry_cache = {}


    # ========================================================
    # TRACE ANCESTRY
    # ========================================================

    print(
        "  Tracing ancestry..."
    )


    all_ancestry = []


    for lineage_number, individual in enumerate(
        starting_individuals
    ):

        branch = trace_ancestry_branch(
            individual["ID"],
            final_generation,
            id_lookup,
            individual_cache,
            ancestry_cache
        )


        for entry in branch:

            new_entry = entry.copy()

            new_entry[
                "Lineage"
            ] = lineage_number


            all_ancestry.append(
                new_entry
            )


    # ========================================================
    # CREATE ANCESTRY DATAFRAME
    # ========================================================

    ancestry_df = pd.DataFrame(
        all_ancestry
    )


    if len(ancestry_df) == 0:

        raise ValueError(
            "No ancestry could be reconstructed."
        )


    # ========================================================
    # CALCULATE COUNTS / PERCENTAGES
    # ========================================================

    patch_proportions = []


    for year in generations:

        year_df = ancestry_df[
            ancestry_df["Year"]
            == year
        ]


        # ----------------------------------------------------
        # UNIQUE MODE
        # ----------------------------------------------------

        if args.unique:

            year_df = year_df.drop_duplicates(
                subset=["ID"]
            )


        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total_ancestors = len(
            year_df
        )


        # ----------------------------------------------------
        # PATCH COUNTS
        # ----------------------------------------------------

        for patch in all_patches:

            patch_count = len(
                year_df[
                    year_df["PatchID"]
                    == patch
                ]
            )


            if total_ancestors > 0:

                proportion = (
                    patch_count
                    / total_ancestors
                )

            else:

                proportion = 0


            patch_proportions.append(
                {
                    "Year": year,

                    "PatchID": patch,

                    "Count": patch_count,

                    "TotalAncestors":
                        total_ancestors,

                    "Percentage":
                        proportion * 100
                }
            )


    proportions_df = pd.DataFrame(
        patch_proportions
    )


    # ========================================================
    # CREATE HEATMAP MATRICES
    # ========================================================

    percentage_matrix = (
        proportions_df.pivot(
            index="Year",
            columns="PatchID",
            values="Percentage"
        )
    )


    percentage_matrix = (
        percentage_matrix
        .fillna(0)
        .reindex(
            index=generations,
            columns=all_patches,
            fill_value=0
        )
    )


    count_matrix = (
        proportions_df.pivot(
            index="Year",
            columns="PatchID",
            values="Count"
        )
    )


    count_matrix = (
        count_matrix
        .fillna(0)
        .reindex(
            index=generations,
            columns=all_patches,
            fill_value=0
        )
    )


    # ========================================================
    # PLOT INDIVIDUAL RUN HEATMAP
    # ========================================================

    if not args.no_plot:

        print(
            "  Creating run heatmap..."
        )


        plot_heatmap(
            percentage_matrix,
            count_matrix,
            generations,
            all_patches,
            run_directory,
            f"MC run {run_directory.name}",
            f"ancestry_heatmap_{args.g}_{args.PID}",
            vmin=0,
            vmax=100
        )


    # ========================================================
    # IMPORTANT MEMORY CLEANUP
    # ========================================================
    #
    # The potentially enormous ancestry information is no
    # longer needed after percentage_matrix/count_matrix
    # have been created.
    #
    # ========================================================

    del ancestry_df
    del all_ancestry
    del ancestry_cache
    del individual_cache
    del id_lookup
    del ind_data
    del final_df
    del starting_individuals


    gc.collect()


    print(
        "  Trajectory data discarded from memory."
    )


    # ========================================================
    # RETURN ONLY SMALL MATRICES
    # ========================================================

    return (
        percentage_matrix.copy(),
        count_matrix.copy(),
        generations,
        all_patches
    )


# ============================================================
# HEATMAP FUNCTION
# ============================================================

def plot_heatmap(
    percentage_matrix,
    count_matrix,
    generations,
    all_patches,
    save_directory,
    title,
    filename,
    vmin=0,
    vmax=100
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
    # COLOUR = PERCENTAGE
    # ========================================================

    image = ax.imshow(
        percentage_matrix.values,
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

            percentage_value = (
                percentage_matrix.iloc[
                    row_number,
                    column_number
                ]
            )


            count_value = (
                count_matrix.iloc[
                    row_number,
                    column_number
                ]
            )


            if percentage_value > 0:

                if args.counts:

                    label = (
                        f"{int(count_value)}"
                    )

                else:

                    label = (
                        f"{percentage_value:.1f}%"
                    )


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
        "Percentage of ancestry"
    )


    if vmax <= 100:

        colourbar.set_ticks(
            [
                0,
                20,
                40,
                60,
                80,
                100
            ]
        )


    plt.tight_layout()


    output_file = (
        save_directory
        / f"{filename}.png"
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
# PROCESS ALL MONTE-CARLO RUNS
# ============================================================

all_percentage_matrices = []

all_count_matrices = []


common_generations = None
common_patches = None


for mc_number, run_directory in run_directories:

    percentage_matrix, count_matrix, generations, patches = (
        process_run(
            run_directory
        )
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
                f"Generation mismatch between Monte-Carlo runs.\n"
                f"Expected: {common_generations}\n"
                f"Found: {generations}\n"
                f"Problem run: {run_directory}"
            )


        if patches != common_patches:

            raise ValueError(
                f"Patch mismatch between Monte-Carlo runs.\n"
                f"Expected: {common_patches}\n"
                f"Found: {patches}\n"
                f"Problem run: {run_directory}"
            )


    # ========================================================
    # ONLY STORE SMALL MATRICES
    # ========================================================

    all_percentage_matrices.append(
        percentage_matrix
    )

    all_count_matrices.append(
        count_matrix
    )


    # ========================================================
    # DELETE REFERENCES TO THIS RUN
    # ========================================================

    del percentage_matrix
    del count_matrix

    gc.collect()


# ============================================================
# COMBINE MONTE-CARLO RUNS
# ============================================================

print()
print(
    "================================================"
)

print(
    "Combining Monte-Carlo runs..."
)

print(
    "================================================"
)


number_runs = len(
    all_percentage_matrices
)


# ============================================================
# SUMMED PERCENTAGE HEATMAP
# ============================================================

summed_percentage_matrix = (
    all_percentage_matrices[0].copy()
)


for matrix in all_percentage_matrices[1:]:

    summed_percentage_matrix = (
        summed_percentage_matrix
        + matrix
    )


# ============================================================
# AVERAGE PERCENTAGE HEATMAP
# ============================================================

average_percentage_matrix = (
    summed_percentage_matrix
    / number_runs
)


# ============================================================
# SUMMED COUNT MATRIX
# ============================================================

summed_count_matrix = (
    all_count_matrices[0].copy()
)


for matrix in all_count_matrices[1:]:

    summed_count_matrix = (
        summed_count_matrix
        + matrix
    )


# ============================================================
# AVERAGE COUNT MATRIX
# ============================================================

average_count_matrix = (
    summed_count_matrix
    / number_runs
)


# ============================================================
# SAVE SUMMED DATA
# ============================================================

summed_output = (
    output_dir
    / f"ancestry_heatmap_SUM_"
      f"{args.g}_{args.PID}.csv"
)


summed_percentage_matrix.to_csv(
    summed_output
)


# ============================================================
# SAVE AVERAGED DATA
# ============================================================

average_output = (
    output_dir
    / f"ancestry_heatmap_AVERAGE_"
      f"{args.g}_{args.PID}.csv"
)


average_percentage_matrix.to_csv(
    average_output
)


print()

print(
    f"Summed percentage data written to:"
)

print(
    f"  {summed_output}"
)


print()

print(
    f"Average percentage data written to:"
)

print(
    f"  {average_output}"
)


# ============================================================
# PLOT SUMMED HEATMAP
# ============================================================

if not args.no_plot:

    print()

    print(
        "Creating summed heatmap..."
    )


    plot_heatmap(
        summed_percentage_matrix,
        summed_count_matrix,
        common_generations,
        common_patches,
        output_dir,
        f"SUMMED spatial ancestry\n"
        f"{args.g} | Patch {args.PID} | "
        f"{number_runs} Monte-Carlo runs",
        f"ancestry_heatmap_SUM_"
        f"{args.g}_{args.PID}",
        vmin=0,
        vmax=100 * number_runs
    )


    # ========================================================
    # PLOT AVERAGE HEATMAP
    # ========================================================

    print()

    print(
        "Creating averaged heatmap..."
    )


    plot_heatmap(
        average_percentage_matrix,
        average_count_matrix,
        common_generations,
        common_patches,
        output_dir,
        f"AVERAGED spatial ancestry\n"
        f"{args.g} | Patch {args.PID} | "
        f"{number_runs} Monte-Carlo runs",
        f"ancestry_heatmap_AVERAGE_"
        f"{args.g}_{args.PID}",
        vmin=0,
        vmax=100
    )


# ============================================================
# CLEAN UP COMBINED MATRICES
# ============================================================

del all_percentage_matrices
del all_count_matrices

gc.collect()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print(
    "================================================"
)

print(
    "Finished."
)

print(
    "================================================"
)

print(
    f"Monte-Carlo runs processed: "
    f"{number_runs}"
)

print(
    f"Starting genotype: "
    f"{args.g}"
)

print(
    f"Starting PatchID: "
    f"{args.PID}"
)

if args.unique:

    print(
        "Counting mode: UNIQUE INDIVIDUALS"
    )

else:

    print(
        "Counting mode: ALL ANCESTRY RECORDS"
    )


if args.counts:

    print(
        "Cell labels: RAW COUNTS"
    )

else:

    print(
        "Cell labels: PERCENTAGES"
    )


print(
    "Heatmap colour: PERCENTAGE"
)

print()

print(
    "Per-run heatmaps are stored inside "
    "their respective Monte-Carlo directories."
)

print(
    "Summed and averaged heatmaps are stored "
    "in the supplied parent directory."
)

print(
    "Large ancestry trajectory data has been "
    "discarded after each run."
)
