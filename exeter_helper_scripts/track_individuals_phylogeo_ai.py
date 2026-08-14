"""
Analyse the spatial ancestry of a CDMetaPOP genotype.

Usage:

    uv run tracking_genotype_ancestry.py \
        -d outputdirectory \
        --genotype "1/2|0/2"

The directory must contain:

    pedigree_tree_output.csv

The script:

1. Reads the pedigree produced by tracking_pedigreetree_ai.py.
2. Constructs a multilocus genotype from:
       L0A0, L0A1, L1A0, L1A1
3. Selects all individuals with the requested genotype.
4. Follows their parents backwards through the pedigree.
5. Records the birth patch of their ancestors.
6. Calculates the distribution of ancestral birth patches at
   each generation backwards.
7. Writes the results to CSV.
8. Produces a heatmap.

Genotype format:

    L0A0/L0A1 | L1A0/L1A1

For example:

    L0A0 = 1
    L0A1 = 2
    L1A0 = 2
    L1A1 = 0

becomes:

    1/2|0/2

The order of alleles within a locus does not matter.

Therefore:

    1/2|0/2

and:

    2/1|2/0

are treated as the same genotype.
"""


import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


# ============================================================
# 1. COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Analyse genotype-conditioned spatial ancestry."
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing pedigree_tree_output.csv"
)

parser.add_argument(
    "--genotype",
    type=str,
    required=True,
    help='Genotype to analyse, e.g. "1/2|0/2"'
)

args = parser.parse_args()

output_dir = Path(args.d)

target_genotype = args.genotype.replace(
    " ",
    ""
)


# ============================================================
# 2. CHECK DIRECTORY
# ============================================================

if not output_dir.exists():

    raise FileNotFoundError(
        f"Directory does not exist: {output_dir}"
    )

if not output_dir.is_dir():

    raise NotADirectoryError(
        f"Not a directory: {output_dir}"
    )


# ============================================================
# 3. FIND PEDIGREE FILE
# ============================================================

pedigree_file = (
    output_dir /
    "pedigree_tree_output.csv"
)


if not pedigree_file.exists():

    raise FileNotFoundError(
        f"Could not find:\n{pedigree_file}"
    )


# ============================================================
# 4. READ PEDIGREE
# ============================================================

print("=" * 70)
print("GENOTYPE-CONDITIONED SPATIAL ANCESTRY")
print("=" * 70)

print(
    f"\nReading pedigree:"
)

print(
    pedigree_file
)

pedigree = pd.read_csv(
    pedigree_file
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "generation",

    "tracking_id",

    "mother_tracking_id",

    "father_tracking_id",

    "birth_patch",

    "L0A0",

    "L0A1",

    "L1A0",

    "L1A1",
]


missing = [

    column
    for column in required_columns
    if column not in pedigree.columns

]


if missing:

    raise ValueError(
        "\nThe pedigree is missing the following "
        f"required columns:\n{missing}"
    )


# ============================================================
# 6. FUNCTION TO CREATE A GENOTYPE
# ============================================================

def make_genotype(
    l0a0,
    l0a1,
    l1a0,
    l1a1
):

    """
    Construct a canonical multilocus genotype.

    Example:

        L0A0 = 2
        L0A1 = 1

        L1A0 = 2
        L1A1 = 0

    becomes:

        1/2|0/2

    Alleles within each locus are sorted so that:

        1/2 == 2/1

    and:

        0/2 == 2/0
    """


    # --------------------------------------------------------
    # Check for missing values
    # --------------------------------------------------------

    values = [
        l0a0,
        l0a1,
        l1a0,
        l1a1
    ]


    if any(pd.isna(value) for value in values):

        return pd.NA


    # --------------------------------------------------------
    # Convert to integers
    # --------------------------------------------------------

    l0a0 = int(l0a0)
    l0a1 = int(l0a1)

    l1a0 = int(l1a0)
    l1a1 = int(l1a1)


    # --------------------------------------------------------
    # Sort alleles within each locus
    # --------------------------------------------------------

    locus_0 = sorted([
        l0a0,
        l0a1
    ])

    locus_1 = sorted([
        l1a0,
        l1a1
    ])


    # --------------------------------------------------------
    # Construct genotype string
    # --------------------------------------------------------

    genotype = (

        f"{locus_0[0]}/{locus_0[1]}"
        "|"
        f"{locus_1[0]}/{locus_1[1]}"

    )


    return genotype


# ============================================================
# 7. CREATE GENOTYPE FOR EVERY INDIVIDUAL
# ============================================================

pedigree["genotype"] = pedigree.apply(

    lambda row: make_genotype(

        row["L0A0"],
        row["L0A1"],
        row["L1A0"],
        row["L1A1"]

    ),

    axis=1
)


# ============================================================
# 8. DISPLAY AVAILABLE GENOTYPES
# ============================================================

genotype_counts = (

    pedigree["genotype"]
    .value_counts()
    .sort_index()

)


print(
    "\nGenotypes found in pedigree:"
)

print(
    genotype_counts
)


# ============================================================
# 9. CHECK TARGET GENOTYPE FORMAT
# ============================================================

if "|" not in target_genotype:

    raise ValueError(

        "\nInvalid genotype format.\n\n"

        "Expected something like:\n"

        "    1/2|0/2\n\n"

        "where the first pair is locus 0 and "
        "the second pair is locus 1."

    )


# ============================================================
# 10. FIND TARGET INDIVIDUALS
# ============================================================

targets = pedigree[

    pedigree["genotype"]
    ==
    target_genotype

]


if len(targets) == 0:

    raise ValueError(

        f"\nNo individuals with genotype "
        f"{target_genotype} were found."

    )


print(
    f"\nTarget genotype: "
    f"{target_genotype}"
)

print(
    f"Number of target individuals: "
    f"{len(targets)}"
)

print(
    f"Generations containing target genotype: "
    f"{targets['generation'].nunique()}"
)


# ============================================================
# CREATE INDIVIDUAL LOOKUP
# ============================================================

"""
tracking_id is the permanent identity of an individual.

Therefore we should NOT use:

    generation + tracking_id

to find parents.

An individual can survive for several generations, so its
record may occur in multiple ind<number>.csv files.

The permanent tracking_id is what identifies the individual.
"""

individual_lookup = {}

for _, row in pedigree.iterrows():

    tracking_id = str(
        row["tracking_id"]
    )

    # Keep the first occurrence of each individual.
    #
    # The individual may occur in multiple yearly files because
    # it survives across generations.

    if tracking_id not in individual_lookup:

        individual_lookup[tracking_id] = row


# ============================================================
# 12. FIND A PARENT
# ============================================================

# ============================================================
# FIND PARENT BY PERMANENT TRACKING ID
# ============================================================

def find_parent(parent_tracking_id):

    """
    Find a parent anywhere in the pedigree.

    We deliberately do NOT specify a generation here.

    This is because CDMetaPOP individuals can survive for
    multiple years. The parent may therefore appear in several
    ind<number>.csv files.

    The tracking_id identifies the biological individual.
    """

    if pd.isna(parent_tracking_id):

        return None


    parent_tracking_id = str(
        parent_tracking_id
    )


    # CDMetaPOP uses -9999 when there is no known parent.

    if parent_tracking_id == "-9999":

        return None


    return individual_lookup.get(
        parent_tracking_id
    )

# ============================================================
# 13. TRACE ANCESTRY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRACING ANCESTRY"
)

print(
    "=" * 70
)


# ancestry_by_generation contains the unique biological
# ancestors at each genealogical distance from the target.
#
# IMPORTANT:
#
# "generation_back" means genealogical distance:
#
#     0 = target individuals
#     1 = parents
#     2 = grandparents
#     3 = great-grandparents
#     ...
#
# It does NOT mean CDMetaPOP simulation generation.


ancestry_by_generation = {}


# ============================================================
# START WITH TARGET INDIVIDUALS
# ============================================================

current_ancestors = {

    str(row["tracking_id"])

    for _, row in targets.iterrows()

}


ancestry_by_generation[0] = (
    current_ancestors.copy()
)


print(
    f"\nGeneration 0: "
    f"{len(current_ancestors)} target individuals"
)


# ============================================================
# FOLLOW PARENTS BACKWARDS
# ============================================================

generation_back = 0


while len(current_ancestors) > 0:

    generation_back += 1


    next_ancestors = set()


    # --------------------------------------------------------
    # Find parents of every current ancestor
    # --------------------------------------------------------

    for tracking_id in current_ancestors:


        individual = individual_lookup.get(
            tracking_id
        )


        if individual is None:

            print(
                f"WARNING: Could not find individual "
                f"{tracking_id}"
            )

            continue


        # ====================================================
        # MOTHER
        # ====================================================

        mother = find_parent(

            individual[
                "mother_tracking_id"
            ]

        )


        if mother is not None:

            next_ancestors.add(

                str(
                    mother["tracking_id"]
                )

            )


        # ====================================================
        # FATHER
        # ====================================================

        father = find_parent(

            individual[
                "father_tracking_id"
            ]

        )


        if father is not None:

            next_ancestors.add(

                str(
                    father["tracking_id"]
                )

            )


    # --------------------------------------------------------
    # Store this genealogical generation
    # --------------------------------------------------------

    ancestry_by_generation[
        generation_back
    ] = next_ancestors


    print(
        f"Generation {generation_back}: "
        f"{len(next_ancestors)} unique ancestors"
    )


    # --------------------------------------------------------
    # Stop when no parents remain
    # --------------------------------------------------------

    if len(next_ancestors) == 0:

        break


    current_ancestors = (
        next_ancestors
    )

# ============================================================
# 14. CALCULATE ANCESTRAL BIRTH-PATCH DISTRIBUTION
# ============================================================

print(
    "\nCalculating ancestral birth-patch distributions..."
)


results = []


for generations_back, ancestors in (
    ancestry_by_generation.items()
):


    if len(ancestors) == 0:

        continue


    ancestor_rows = []


    for tracking_id in ancestors:


        individual = individual_lookup.get(
            tracking_id
        )


        if individual is not None:

            ancestor_rows.append(
                individual
            )


    if len(ancestor_rows) == 0:

        continue


    ancestor_df = pd.DataFrame(
        ancestor_rows
    )


    # --------------------------------------------------------
    # Count birth patches
    # --------------------------------------------------------

    counts = (

        ancestor_df[
            "birth_patch"
        ]
        .value_counts()
        .sort_index()

    )


    total = counts.sum()


    proportions = (
        counts / total
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    for patch, proportion in (
        proportions.items()
    ):

        results.append({

            "genotype":
                target_genotype,

            "generations_back":
                generations_back,

            "birth_patch":
                patch,

            "n_unique_ancestors":
                counts.loc[patch],

            "proportion":
                proportion,

        })

# ============================================================
# 15. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


if len(results_df) == 0:

    raise ValueError(
        "No ancestral birth-patch data "
        "could be generated."
    )


results_df = results_df.sort_values(

    [
        "generations_back",
        "birth_patch"
    ]

)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

# Make genotype safe to use in a filename.
#
# For example:
#
#     1/1|1/1
#
# becomes:
#
#     1-1_1-1
#
# "/" cannot appear directly in the filename because Linux
# interprets it as a directory separator.

safe_genotype = target_genotype.replace(
    "/",
    "-"
).replace(
    "|",
    "_"
)


output_file = (

    output_dir /
    f"ancestral_birth_patch_{safe_genotype}.csv"

)


results_df.to_csv(

    output_file,

    index=False

)


print(
    "\nAncestral distribution written to:"
)

print(
    output_file
)


# ============================================================
# 17. CREATE HEATMAP
# ============================================================

heatmap = results_df.pivot(

    index="generations_back",

    columns="birth_patch",

    values="proportion"

)


heatmap = heatmap.sort_index(
    ascending=True
)


plt.figure(
    figsize=(12, 8)
)


plt.imshow(

    heatmap,

    aspect="auto",

    interpolation="nearest"

)


plt.colorbar(

    label="Proportion of unique ancestors"

)


plt.xlabel(
    "Birth patch"
)


plt.ylabel(
    "Generations back"
)


plt.title(

    f"Ancestral birth-patch distribution\n"
    f"Genotype: {target_genotype}"

)


plt.xticks(

    range(len(heatmap.columns)),

    heatmap.columns

)


plt.yticks(

    range(len(heatmap.index)),

    heatmap.index

)


plt.tight_layout()


figure_file = (

    output_dir /
    f"ancestral_birth_patch_{safe_genotype}.png"

)


plt.savefig(

    figure_file,

    dpi=300

)


plt.close()


print(
    "Heatmap written to:"
)

print(
    figure_file
)


# ============================================================
# 18. FINISHED
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)