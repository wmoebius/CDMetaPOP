import pandas as pd
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Track the spatial ancestry of a genotype in a patch."
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing ind<number>.csv files"
)

parser.add_argument(
    "-g",
    type=str,
    required=True,
    choices=[
        "L0A0A0",
        "L0A0A1",
        "L0A1A1",
        "L1A0A0",
        "L1A0A1",
        "L1A1A1"
    ],
    help="The genotype in question"
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
    help="Do not create the heatmap"
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
# FIND INDIVIDUAL FILES
# ============================================================

ind_files = []

for file in output_dir.glob("ind*.csv"):

    match = re.fullmatch(
        r"ind(\d+)\.csv",
        file.name
    )

    if match:

        generation = int(
            match.group(1)
        )

        ind_files.append(
            (generation, file)
        )


ind_files.sort(
    key=lambda x: x[0]
)


if len(ind_files) == 0:

    raise FileNotFoundError(
        f"No ind<number>.csv files found in "
        f"{output_dir}"
    )


generations = []

for generation, file in ind_files:

    generations.append(
        generation
    )


final_generation = generations[-1]


# ============================================================
# READ ALL IND FILES
# ============================================================

print()

print(
    "Reading individual files..."
)

ind_data = {}

for generation, file in ind_files:

    print(
        f"  Reading generation {generation}: "
        f"{file.name}"
    )

    df = pd.read_csv(
        file
    )

    ind_data[
        generation
    ] = df


print()

print(
    f"Final generation: "
    f"{final_generation}"
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
# CREATE ID LOOKUP TABLES
# ============================================================

print()

print(
    "Creating ID lookup tables..."
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


# ============================================================
# FIND STARTING INDIVIDUALS
# ============================================================

print()

print(
    "Finding starting individuals..."
)

final_df = ind_data[
    final_generation
]

starting_individuals = []


for _, row in final_df.iterrows():

    if str(
        row["PatchID"]
    ) != args.PID:

        continue


    if genotype_matches(
        row,
        args.g
    ):

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
    f"Found {len(starting_individuals)} "
    f"starting individuals"
)


if len(starting_individuals) == 0:

    raise ValueError(
        f"No individuals with genotype "
        f"{args.g} were found in "
        f"PatchID {args.PID}"
    )


# ============================================================
# CACHES
# ============================================================

individual_cache = {}

ancestry_cache = {}


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
    start_generation
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

        print(
            f"        Looking for "
            f"{individual_id} "
            f"in generation "
            f"{current_generation}"
        )


        generation_lookup = id_lookup.get(
            current_generation,
            {}
        )


        # ----------------------------------------------------
        # INDIVIDUAL HAS DISAPPEARED
        # ----------------------------------------------------

        if individual_id not in generation_lookup:

            print(
                f"        {individual_id} "
                f"NOT FOUND in generation "
                f"{current_generation}"
            )

            break


        # ----------------------------------------------------
        # INDIVIDUAL EXISTS
        # ----------------------------------------------------

        row = generation_lookup[
            individual_id
        ]


        print(
            f"        FOUND {individual_id} "
            f"in generation "
            f"{current_generation} "
            f"in patch "
            f"{row['PatchID']}"
        )


        # ----------------------------------------------------
        # Record this individual's location
        # ----------------------------------------------------

        history.append(
            {
                "Year": current_generation,
                "ID": individual_id,
                "PatchID": row["PatchID"]
            }
        )


        # ----------------------------------------------------
        # Store parents from the most recent occurrence
        # ----------------------------------------------------

        mother_id = get_last_three(
            row["MID"]
        )

        father_id = get_last_three(
            row["FID"]
        )


        # ----------------------------------------------------
        # Move backwards one generation
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
    start_generation
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

        print(
            f"      Using cached branch for "
            f"{individual_id} "
            f"from generation "
            f"{start_generation}"
        )

        return ancestry_cache[
            cache_key
        ]


    ancestry = []


    # ========================================================
    # TRACE THIS INDIVIDUAL
    # ========================================================

    result = trace_individual(
        individual_id,
        start_generation
    )


    # --------------------------------------------------------
    # Add this individual's history
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


    print(
        f"      {individual_id} "
        f"disappeared before generation "
        f"{parent_start_generation + 1}"
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


        print(
            f"      Following mother "
            f"{mother_id} "
            f"from generation "
            f"{parent_start_generation}"
        )


        mother_branch = trace_ancestry_branch(
            mother_id,
            parent_start_generation
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


        print(
            f"      Following father "
            f"{father_id} "
            f"from generation "
            f"{parent_start_generation}"
        )


        father_branch = trace_ancestry_branch(
            father_id,
            parent_start_generation
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
# TRACE ALL STARTING INDIVIDUALS
# ============================================================

print()

print(
    "Tracing ancestry..."
)


all_ancestry = []


number_starting = len(
    starting_individuals
)


for lineage_number, individual in enumerate(
    starting_individuals
):

    print()
    print(
        "================================================"
    )

    print(
        f"Individual "
        f"{lineage_number + 1} / "
        f"{number_starting}"
    )

    print(
        f"ID: "
        f"{individual['ID']}"
    )

    print(
        "================================================"
    )


    branch = trace_ancestry_branch(
        individual["ID"],
        final_generation
    )


    # --------------------------------------------------------
    # Add lineage number
    # --------------------------------------------------------

    for entry in branch:

        new_entry = entry.copy()

        new_entry[
            "Lineage"
        ] = lineage_number


        all_ancestry.append(
            new_entry
        )


# ============================================================
# CREATE ANCESTRY DATAFRAME
# ============================================================

ancestry_df = pd.DataFrame(
    all_ancestry
)


if len(ancestry_df) == 0:

    raise ValueError(
        "No ancestry could be reconstructed."
    )


ancestry_output = (
    output_dir
    / f"ancestry_{args.g}_{args.PID}.csv"
)


ancestry_df.to_csv(
    ancestry_output,
    index=False
)


print()

print(
    "Ancestry data written to:"
)

print(
    f"  {ancestry_output}"
)


# ============================================================
# DIAGNOSTIC: NUMBER OF RECORDS PER GENERATION
# ============================================================

print()

print(
    "Ancestry records by generation:"
)


for generation in generations:

    generation_df = ancestry_df[
        ancestry_df["Year"]
        == generation
    ]


    record_count = len(
        generation_df
    )


    unique_count = generation_df[
        "ID"
    ].nunique()


    print(
        f"  Generation {generation}: "
        f"{record_count} records, "
        f"{unique_count} unique individuals"
    )


# ============================================================
# FIND ALL PATCHES
# ============================================================

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


# ============================================================
# CALCULATE ANCESTRAL COUNTS
# ============================================================
#
# By default:
#
#     Every occurrence in the ancestry tree is counted.
#
# With --unique:
#
#     Each individual is counted only once within a
#     particular generation.
#
# ============================================================

print()

if args.unique:

    print(
        "Calculating UNIQUE ancestral counts..."
    )

else:

    print(
        "Calculating ancestral counts..."
    )


patch_proportions = []


for year in generations:

    year_df = ancestry_df[
        ancestry_df["Year"]
        == year
    ]


    # ========================================================
    # UNIQUE MODE
    # ========================================================

    if args.unique:

        # ----------------------------------------------------
        # Remove duplicate occurrences of the same individual
        # within this generation.
        # ----------------------------------------------------

        year_df = year_df.drop_duplicates(
            subset=["ID"]
        )


    # ========================================================
    # TOTAL NUMBER OF ANCESTORS
    # ========================================================

    total_ancestors = len(
        year_df
    )


    # ========================================================
    # COUNT EACH PATCH
    # ========================================================

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

                "Proportion":
                    proportion,

                "Percentage":
                    proportion * 100
            }
        )


proportions_df = pd.DataFrame(
    patch_proportions
)


# ============================================================
# SAVE PROPORTIONS
# ============================================================

if args.unique:

    unique_suffix = "unique"

else:

    unique_suffix = "all"


proportions_output = (
    output_dir
    / f"ancestry_proportions_"
      f"{args.g}_{args.PID}_"
      f"{unique_suffix}.csv"
)


proportions_df.to_csv(
    proportions_output,
    index=False
)


print()

print(
    "Ancestral proportions written to:"
)

print(
    f"  {proportions_output}"
)


# ============================================================
# CREATE PERCENTAGE HEATMAP DATA
# ============================================================
#
# IMPORTANT:
#
# The HEATMAP COLOUR is ALWAYS based on percentage.
#
# --counts only changes the text displayed inside the cells.
#
# ============================================================

heatmap_df = proportions_df.pivot(
    index="Year",
    columns="PatchID",
    values="Percentage"
)


heatmap_df = heatmap_df.fillna(
    0
)


# ------------------------------------------------------------
# Explicitly include every generation and patch
# ------------------------------------------------------------

heatmap_df = heatmap_df.reindex(
    index=generations,
    columns=all_patches,
    fill_value=0
)


# ============================================================
# SAVE HEATMAP DATA
# ============================================================

heatmap_data_output = (
    output_dir
    / f"ancestry_heatmap_data_"
      f"{args.g}_{args.PID}_"
      f"{unique_suffix}.csv"
)


heatmap_df.to_csv(
    heatmap_data_output
)


print()

print(
    "Heatmap percentage data written to:"
)

print(
    f"  {heatmap_data_output}"
)


# ============================================================
# CREATE COUNT DATA FOR ANNOTATIONS
# ============================================================

count_heatmap_df = proportions_df.pivot(
    index="Year",
    columns="PatchID",
    values="Count"
)


count_heatmap_df = count_heatmap_df.fillna(
    0
)


count_heatmap_df = count_heatmap_df.reindex(
    index=generations,
    columns=all_patches,
    fill_value=0
)


# ============================================================
# PLOT HEATMAP
# ============================================================

if not args.no_plot:

    print()

    print(
        "Creating heatmap..."
    )


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
    # HEATMAP COLOUR
    # ========================================================
    #
    # ALWAYS USE PERCENTAGE.
    #
    # This means that --counts has NO effect on the colour.
    #
    # ========================================================

    image = ax.imshow(
        heatmap_df.values,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=100
    )


    # ========================================================
    # LABELS
    # ========================================================

    ax.set_xlabel(
        "Patch ID"
    )

    ax.set_ylabel(
        "Generation"
    )


    if args.unique:

        ancestry_type = (
            "Unique ancestors"
        )

    else:

        ancestry_type = (
            "Ancestry"
        )


    if args.counts:

        display_type = (
            "Counts shown; colour = percentage"
        )

    else:

        display_type = (
            "Percentage"
        )


    ax.set_title(
        f"Spatial ancestry of "
        f"{args.g} in Patch {args.PID}\n"
        f"({ancestry_type}; {display_type})"
    )


    # ========================================================
    # X AXIS = PATCH
    # ========================================================

    ax.set_xticks(
        range(
            len(all_patches)
        )
    )

    ax.set_xticklabels(
        all_patches
    )


    # ========================================================
    # Y AXIS = GENERATION
    # ========================================================

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
                heatmap_df.iloc[
                    row_number,
                    column_number
                ]
            )


            count_value = (
                count_heatmap_df.iloc[
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
    # COLORBAR
    # ========================================================

    colourbar = fig.colorbar(
        image,
        ax=ax
    )


    colourbar.set_label(
        "Percentage of ancestry"
    )


    # --------------------------------------------------------
    # Explicitly force colourbar to 0--100%
    # --------------------------------------------------------

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


    heatmap_output = (
        output_dir
        / f"ancestry_heatmap_"
          f"{args.g}_{args.PID}_"
          f"{unique_suffix}.png"
    )


    plt.savefig(
        heatmap_output,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print()

    print(
        "Heatmap written to:"
    )

    print(
        f"  {heatmap_output}"
    )


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
    f"Starting individuals: "
    f"{len(starting_individuals)}"
)

print(
    f"Cached individual histories: "
    f"{len(individual_cache)}"
)

print(
    f"Cached ancestry branches: "
    f"{len(ancestry_cache)}"
)

print(
    f"Ancestry records: "
    f"{len(ancestry_df)}"
)

print()

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
    "Output files:"
)

print(
    f"  {ancestry_output}"
)

print(
    f"  {proportions_output}"
)

print(
    f"  {heatmap_data_output}"
)

if not args.no_plot:

    print(
        f"  {heatmap_output}"
    )