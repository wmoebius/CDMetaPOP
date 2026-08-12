"""
Construct and visualise a pedigree tree from CDMetaPOP ind*.csv files.

Usage:

    uv run tracking_pedigreetree_ai.py -d outputdirectory

The script:

1. Finds ind0.csv, ind2.csv, ind3.csv, etc.
2. Ignores indSample*.csv and other files that do not match ind<number>.csv.
3. Uses the last three fields of ID as the permanent individual identifier.
4. Uses MID and FID to identify the mother and father.
5. Constructs a directed pedigree graph:
       Mother ----\
                   ---> Offspring
       Father ----/
6. Places individuals according to their generation.
7. Visualises the pedigree using matplotlib.

The last three fields are assumed to uniquely identify an individual,
as established by the accompanying ID-checking analysis.
"""


import pandas as pd
from pathlib import Path
import re
import argparse

import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# 1. SETTINGS
# ============================================================

parser = argparse.ArgumentParser(
    description="Construct and visualise a CDMetaPOP pedigree tree."
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing ind<number>.csv files"
)

args = parser.parse_args()

output_dir = Path(args.d)


# ============================================================
# 2. CHECK THAT THE DIRECTORY EXISTS
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
# 3. FIND THE INDIVIDUAL FILES
# ============================================================

ind_files = []


for file in output_dir.glob("ind*.csv"):

    # This matches:
    #
    #     ind0.csv
    #     ind1.csv
    #     ind2.csv
    #     ind10.csv
    #
    # but NOT:
    #
    #     indSample1.csv
    #     indSample2.csv

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


# Sort numerically by generation

ind_files.sort(
    key=lambda x: x[0]
)


if len(ind_files) == 0:

    raise FileNotFoundError(
        f"No files matching ind<number>.csv "
        f"were found in {output_dir}"
    )


print("=" * 70)
print("CDMetaPOP PEDIGREE")
print("=" * 70)

print("\nFiles being analysed:")

for generation, file in ind_files:

    print(
        f"Generation {generation}: "
        f"{file.name}"
    )


# ============================================================
# 4. FUNCTION FOR EXTRACTING THE INDIVIDUAL ID
# ============================================================

def get_last_three(id_string):

    """
    Extract the last three underscore-separated
    fields from a CDMetaPOP ID.

    Example:

        R1_F1_m1f1_P1_Y0_UO4

    becomes:

        P1_Y0_UO4
    """

    parts = str(id_string).split("_")

    return "_".join(
        parts[-3:]
    )


# ============================================================
# 5. READ ALL GENERATIONS
# ============================================================

all_records = []


for generation, file in ind_files:

    print(
        f"\nReading generation {generation}: "
        f"{file.name}"
    )

    df = pd.read_csv(file)


    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "ID",
        "MID",
        "FID"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{file.name} is missing required "
            f"columns: {missing}"
        )


    # --------------------------------------------------------
    # Construct permanent IDs
    # --------------------------------------------------------

    df["tracking_id"] = df["ID"].apply(
        get_last_three
    )

    df["mother_tracking_id"] = df["MID"].apply(
        get_last_three
    )

    df["father_tracking_id"] = df["FID"].apply(
        get_last_three
    )


    # --------------------------------------------------------
    # Store the information needed for the pedigree
    # --------------------------------------------------------

    for _, row in df.iterrows():

        all_records.append({

            "generation": generation,

            "tracking_id": row["tracking_id"],

            "ID": row["ID"],

            "MID": row["MID"],

            "FID": row["FID"],

            "mother_tracking_id":
                row["mother_tracking_id"],

            "father_tracking_id":
                row["father_tracking_id"],

            "sex": row["sex"]
                if "sex" in df.columns
                else None,

            "PatchID": row["PatchID"]
                if "PatchID" in df.columns
                else None
        })


# ============================================================
# 6. CONVERT TO DATAFRAME
# ============================================================

individuals = pd.DataFrame(
    all_records
)


print("\n")
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"Generations: "
    f"{len(ind_files)}"
)

print(
    f"Individual records: "
    f"{len(individuals)}"
)

print(
    f"Unique individuals: "
    f"{individuals['tracking_id'].nunique()}"
)


# ============================================================
# 7. CHECK FOR DUPLICATE INDIVIDUALS
# ============================================================

duplicates = individuals[
    individuals.duplicated(
        "tracking_id",
        keep=False
    )
]


if len(duplicates) > 0:

    print("\nWARNING:")
    print(
        "Some tracking IDs occur more than once "
        "in the dataset."
    )

    print(
        duplicates[
            [
                "generation",
                "tracking_id",
                "ID"
            ]
        ]
    )


# ============================================================
# 8. CREATE THE PEDIGREE GRAPH
# ============================================================

G = nx.DiGraph()


# ------------------------------------------------------------
# Add every individual as a node
# ------------------------------------------------------------

for _, row in individuals.iterrows():

    G.add_node(

        row["tracking_id"],

        generation=row["generation"],

        ID=row["ID"],

        sex=row["sex"],

        PatchID=row["PatchID"]
    )


# ------------------------------------------------------------
# Add parent → offspring edges
# ------------------------------------------------------------

for _, row in individuals.iterrows():

    child = row["tracking_id"]

    mother = row["mother_tracking_id"]

    father = row["father_tracking_id"]


    # --------------------------------------------------------
    # Mother
    # --------------------------------------------------------

    # CDMetaPOP uses -1 in the example when
    # there is no known mother.

    if (
        mother not in ["-1", "nan", "None"]
        and mother != child
    ):

        G.add_edge(
            mother,
            child,
            parent="mother"
        )


    # --------------------------------------------------------
    # Father
    # --------------------------------------------------------

    if (
        father not in ["-1", "nan", "None"]
        and father != child
    ):

        G.add_edge(
            father,
            child,
            parent="father"
        )


print("\n")
print("=" * 70)
print("PEDIGREE GRAPH")
print("=" * 70)

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Parent-child relationships: "
    f"{G.number_of_edges()}"
)


# ============================================================
# 9. FIND PARENTS THAT ARE NOT IN THE DATA
# ============================================================

missing_parents = set()


for _, row in individuals.iterrows():

    mother = row["mother_tracking_id"]
    father = row["father_tracking_id"]


    if (
        mother not in ["-1", "nan", "None"]
        and mother not in G.nodes
    ):

        missing_parents.add(
            mother
        )


    if (
        father not in ["-1", "nan", "None"]
        and father not in G.nodes
    ):

        missing_parents.add(
            father
        )


if missing_parents:

    print(
        f"\nParents referenced but not present "
        f"in the supplied files: "
        f"{len(missing_parents)}"
    )


    # Add them as placeholder nodes

    for parent in missing_parents:

        G.add_node(
            parent,
            generation=None,
            missing=True
        )


# ============================================================
# 10. CREATE A GENERATION-BASED LAYOUT
# ============================================================

# We want:
#
#       Generation 0
#            ↑
#       Generation 1
#            ↑
#       Generation 2
#            ↑
#       Generation 3
#
# So generation determines the vertical position.


generations = {}


for node, data in G.nodes(data=True):

    generation = data.get(
        "generation"
    )


    if generation is None:

        continue


    if generation not in generations:

        generations[generation] = []


    generations[generation].append(
        node
    )


# ------------------------------------------------------------
# Assign x positions
# ------------------------------------------------------------

pos = {}


for generation in sorted(
    generations.keys()
):

    nodes = generations[generation]


    # Sort by tracking ID so that
    # the layout is reproducible.

    nodes = sorted(nodes)


    n = len(nodes)


    for i, node in enumerate(nodes):

        if n == 1:

            x = 0

        else:

            x = (
                i - (n - 1) / 2
            )


        pos[node] = (
            x,
            generation
        )


# ------------------------------------------------------------
# Place missing parents above their offspring
# ------------------------------------------------------------

missing_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("generation") is None
]


for node in missing_nodes:

    children = list(
        G.successors(node)
    )


    if len(children) > 0:

        child_positions = [
            pos[child][0]
            for child in children
            if child in pos
        ]


        if child_positions:

            x = sum(
                child_positions
            ) / len(
                child_positions
            )

            child_generations = [
                pos[child][1]
                for child in children
                if child in pos
            ]

            y = min(
                child_generations
            ) - 1

            pos[node] = (
                x,
                y
            )


# ============================================================
# 11. CREATE THE FIGURE
# ============================================================

number_of_nodes = G.number_of_nodes()


# Make the figure reasonably large for
# large pedigrees.

width = max(
    12,
    len(generations) * 2
)

height = max(
    8,
    max(
        len(nodes)
        for nodes in generations.values()
    ) / 10
)


fig, ax = plt.subplots(
    figsize=(width, height)
)


# ============================================================
# 12. DRAW EDGES
# ============================================================

nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    arrows=True,
    arrowsize=10,
    alpha=0.4,
    width=0.8
)


# ============================================================
# 13. DRAW NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_size=35,
    alpha=0.8
)


# ============================================================
# 14. LABEL NODES
# ============================================================

# For a large simulation, labelling every node would
# make the plot unreadable.
#
# Therefore labels are only displayed when there
# are relatively few individuals.

if number_of_nodes <= 150:

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=6
    )


# ============================================================
# 15. LABEL GENERATIONS
# ============================================================

for generation in sorted(
    generations.keys()
):

    ax.axhline(
        generation,
        alpha=0.15,
        linewidth=0.8
    )

    ax.text(
        -max(
            1,
            len(
                generations[generation]
            ) / 2
        ),
        generation,
        f"Generation {generation}",
        verticalalignment="center",
        fontsize=9
    )


# ============================================================
# 16. FORMAT THE PLOT
# ============================================================

ax.set_title(
    "CDMetaPOP Pedigree"
)

ax.set_xlabel(
    "Individuals"
)

ax.set_ylabel(
    "Generation"
)

ax.set_yticks(
    sorted(
        generations.keys()
    )
)

ax.set_yticklabels(
    [
        f"Generation {g}"
        for g in sorted(
            generations.keys()
        )
    ]
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.tight_layout()


# ============================================================
# 17. SAVE FIGURE
# ============================================================

output_file = (
    output_dir /
    "pedigree_tree.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


print(
    f"\nPedigree saved to:"
)

print(
    output_file
)


# ============================================================
# 18. SHOW FIGURE
# ============================================================

plt.show()
