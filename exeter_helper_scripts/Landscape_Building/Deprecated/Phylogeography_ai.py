import pandas as pd
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt


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
    "--no-plot",
    action="store_true",
    help="Do not create the heatmap"
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
print("Reading individual files...")

ind_data = {}

for generation, file in ind_files:

    print(
        f"  Reading generation {generation}: "
        f"{file.name}"
    )

    df = pd.read_csv(file)

    ind_data[generation] = df


print()
print(
    f"Final generation: {final_generation}"
)


# ============================================================
# EXTRACT BIRTH GENERATION FROM ID
# ============================================================

def get_birth_generation(
    individual_id
):

    """
    Extract the Y<number> section of a CDMetaPOP ID.

    Example:

        RD1_F3_m3f2_P3_Y8_UO52

    returns:

        8
    """

    match = re.search(
        r"_Y(\d+)_",
        str(individual_id)
    )

    if match is None:

        return None

    return int(
        match.group(1)
    )


# ============================================================
# FUNCTION TO EXTRACT LAST THREE ID FIELDS
# ============================================================

def get_last_three(id_string):

    parts = str(id_string).split("_")

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
print("Creating ID lookup tables...")

id_lookup = {}

for generation in generations:

    id_lookup[generation] = {}

    df = ind_data[generation]

    for _, row in df.iterrows():

        individual_id = str(
            row["ID"]
        )

        id_lookup[generation][
            individual_id
        ] = row


# ============================================================
# FIND STARTING INDIVIDUALS
# ============================================================

print()
print("Finding starting individuals...")

final_df = ind_data[final_generation]

starting_individuals = []

for _, row in final_df.iterrows():

    if str(row["PatchID"]) != args.PID:

        continue

    if genotype_matches(
        row,
        args.g
    ):

        individual = {
            "ID": str(row["ID"]),
            "PatchID": row["PatchID"],
            "MID": row["MID"],
            "FID": row["FID"]
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
        f"No individuals with genotype {args.g} "
        f"were found in PatchID {args.PID}"
    )


# ============================================================
# CACHE
# ============================================================

individual_cache = {}

ancestry_cache = {}


# ============================================================
# CHECK PARENT ID
# ============================================================

def valid_parent_id(
    parent_id
):

    if pd.isna(parent_id):

        return False

    if str(parent_id) == "":

        return False

    if str(parent_id).lower() == "nan":

        return False

    return True


# ============================================================
# TRACE ONE INDIVIDUAL
# ============================================================
#
# This is now based on the birth year contained in the ID.
#
# We do NOT assume that:
#
#     birth year = starting year - 1
#
# or that parents have the same birth year.
#
# ============================================================

def trace_individual(
    individual_id,
    start_generation
):

    individual_id = str(
        individual_id
    )


    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    if individual_id in individual_cache:

        return individual_cache[
            individual_id
        ]


    history = []


    mother_id = None
    father_id = None


    # --------------------------------------------------------
    # Determine birth generation from ID
    # --------------------------------------------------------

    birth_generation = get_birth_generation(
        individual_id
    )


    # --------------------------------------------------------
    # If we cannot determine birth year from the ID,
    # fall back to searching the files.
    # --------------------------------------------------------

    if birth_generation is None:

        search_generations = []

        for generation in generations:

            if generation <= start_generation:

                search_generations.append(
                    generation
                )

    else:

        search_generations = []

        for generation in generations:

            if (
                generation <= start_generation
                and generation >= birth_generation
            ):

                search_generations.append(
                    generation
                )


    # --------------------------------------------------------
    # Search from latest to earliest
    # --------------------------------------------------------

    search_generations.reverse()


    for generation in search_generations:

        generation_lookup = id_lookup[
            generation
        ]


        if individual_id not in generation_lookup:

            continue


        row = generation_lookup[
            individual_id
        ]


        history.append(
            {
                "Year": generation,
                "ID": individual_id,
                "PatchID": row["PatchID"]
            }
        )


        # ----------------------------------------------------
        # Keep the parents from the individual's birth record.
        #
        # Once we reach the birth year, stop.
        # ----------------------------------------------------

        if (
            birth_generation is not None
            and generation == birth_generation
        ):

            mother_id = row["MID"]
            father_id = row["FID"]

            break


        # ----------------------------------------------------
        # If no birth generation was available, keep updating
        # the parent IDs while searching backwards.
        # ----------------------------------------------------

        if birth_generation is None:

            mother_id = row["MID"]
            father_id = row["FID"]


    result = {
        "History": history,
        "Mother": mother_id,
        "Father": father_id,
        "BirthGeneration": birth_generation
    }


    individual_cache[
        individual_id
    ] = result


    return result


# ============================================================
# TRACE COMPLETE ANCESTRY BRANCH
# ============================================================

def trace_ancestry_branch(
    individual_id,
    start_generation
):

    individual_id = str(
        individual_id
    )


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if individual_id in ancestry_cache:

        return ancestry_cache[
            individual_id
        ]


    ancestry = []


    # --------------------------------------------------------
    # Trace this individual
    # --------------------------------------------------------

    result = trace_individual(
        individual_id,
        start_generation
    )


    # --------------------------------------------------------
    # Add individual's own history
    # --------------------------------------------------------

    for entry in result["History"]:

        ancestry.append(
            entry.copy()
        )


    # --------------------------------------------------------
    # Find the birth generation
    # --------------------------------------------------------

    birth_generation = result[
        "BirthGeneration"
    ]


    if birth_generation is None:

        parent_generation = None

    else:

        parent_generation = (
            birth_generation - 1
        )


    # --------------------------------------------------------
    # If this individual was born in generation 0,
    # there are no earlier generations to trace.
    # --------------------------------------------------------

    if parent_generation is None:

        ancestry_cache[
            individual_id
        ] = ancestry

        return ancestry


    if parent_generation < 0:

        ancestry_cache[
            individual_id
        ] = ancestry

        return ancestry


    # --------------------------------------------------------
    # MOTHER
    # --------------------------------------------------------

    mother_id = result[
        "Mother"
    ]


    if valid_parent_id(
        mother_id
    ):

        mother_branch = trace_ancestry_branch(
            mother_id,
            parent_generation
        )


        for entry in mother_branch:

            ancestry.append(
                entry.copy()
            )


    # --------------------------------------------------------
    # FATHER
    # --------------------------------------------------------

    father_id = result[
        "Father"
    ]


    if valid_parent_id(
        father_id
    ):

        father_branch = trace_ancestry_branch(
            father_id,
            parent_generation
        )


        for entry in father_branch:

            ancestry.append(
                entry.copy()
            )


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    ancestry_cache[
        individual_id
    ] = ancestry


    return ancestry


# ============================================================
# TRACE ALL STARTING INDIVIDUALS
# ============================================================

print()
print("Tracing ancestry...")

all_ancestry = []

number_starting = len(
    starting_individuals
)


for lineage_number, individual in enumerate(
    starting_individuals
):

    print(
        f"  Individual "
        f"{lineage_number + 1} / "
        f"{number_starting}: "
        f"{get_last_three(individual['ID'])}"
    )


    branch = trace_ancestry_branch(
        individual["ID"],
        final_generation
    )


    for entry in branch:

        new_entry = entry.copy()

        new_entry["Lineage"] = (
            lineage_number
        )

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
    f"Ancestry data written to:"
)

print(
    f"  {ancestry_output}"
)


# ============================================================
# DIAGNOSTIC: NUMBER OF ANCESTRY RECORDS PER GENERATION
# ============================================================

print()
print(
    "Ancestry records by generation:"
)


for generation in generations:

    count = len(
        ancestry_df[
            ancestry_df["Year"] == generation
        ]
    )

    print(
        f"  Generation {generation}: "
        f"{count}"
    )


# ============================================================
# CALCULATE ALL PATCHES
# ============================================================

all_patches = set()

for generation in generations:

    df = ind_data[
        generation
    ]

    for patch in df["PatchID"].unique():

        all_patches.add(
            patch
        )


all_patches = sorted(
    all_patches
)


# ============================================================
# CALCULATE ANCESTRAL PROPORTIONS
# ============================================================

print()
print(
    "Calculating ancestral proportions..."
)


patch_proportions = []


for year in generations:

    year_df = ancestry_df[
        ancestry_df["Year"] == year
    ]


    total_ancestors = len(
        year_df
    )


    for patch in all_patches:

        patch_count = len(
            year_df[
                year_df["PatchID"] == patch
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
                "Proportion": proportion,
                "Percentage": (
                    proportion * 100
                )
            }
        )


proportions_df = pd.DataFrame(
    patch_proportions
)


# ============================================================
# SAVE PROPORTIONS
# ============================================================

proportions_output = (
    output_dir
    / f"ancestry_proportions_{args.g}_{args.PID}.csv"
)


proportions_df.to_csv(
    proportions_output,
    index=False
)


print(
    f"Ancestral proportions written to:"
)

print(
    f"  {proportions_output}"
)


# ============================================================
# CREATE HEATMAP DATA
# ============================================================

heatmap_df = proportions_df.pivot(
    index="Year",
    columns="PatchID",
    values="Percentage"
)


heatmap_df = heatmap_df.fillna(
    0
)


# Explicitly force every generation to appear

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
    / f"ancestry_heatmap_data_{args.g}_{args.PID}.csv"
)


heatmap_df.to_csv(
    heatmap_data_output
)


print(
    f"Heatmap data written to:"
)

print(
    f"  {heatmap_data_output}"
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


    image = ax.imshow(
        heatmap_df.values,
        aspect="auto",
        interpolation="nearest",
        origin="upper"
    )


    # --------------------------------------------------------
    # AXIS LABELS
    # --------------------------------------------------------

    ax.set_xlabel(
        "Patch ID"
    )

    ax.set_ylabel(
        "Generation"
    )

    ax.set_title(
        f"Spatial ancestry of {args.g} "
        f"in Patch {args.PID}"
    )


    # --------------------------------------------------------
    # X AXIS = PATCH
    # --------------------------------------------------------

    ax.set_xticks(
        range(
            len(all_patches)
        )
    )

    ax.set_xticklabels(
        all_patches
    )


    # --------------------------------------------------------
    # Y AXIS = GENERATION
    # --------------------------------------------------------

    ax.set_yticks(
        range(
            len(generations)
        )
    )

    ax.set_yticklabels(
        generations
    )


    # --------------------------------------------------------
    # PERCENTAGE LABELS
    # --------------------------------------------------------

    for row_number in range(
        len(generations)
    ):

        for column_number in range(
            len(all_patches)
        ):

            value = heatmap_df.iloc[
                row_number,
                column_number
            ]


            if value > 0:

                ax.text(
                    column_number,
                    row_number,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8
                )


    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    colourbar = fig.colorbar(
        image,
        ax=ax
    )


    colourbar.set_label(
        "Percentage of ancestry"
    )


    plt.tight_layout()


    heatmap_output = (
        output_dir
        / f"ancestry_heatmap_{args.g}_{args.PID}.png"
    )


    plt.savefig(
        heatmap_output,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Heatmap written to:"
    )

    print(
        f"  {heatmap_output}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "Finished."
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