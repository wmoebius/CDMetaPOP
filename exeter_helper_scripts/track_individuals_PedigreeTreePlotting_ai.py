"""
Plot pedigrees for n individuals from the final generation.

Usage:

    uv run tracking_pedigree_plotting_ai.py -d outputdirectory

The value of n is set below.

For each selected individual, the script traces their ancestry
back through all available generations and creates a separate
pedigree plot.
"""


import pandas as pd
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# 1. SETTINGS
# ============================================================

# Number of final-generation individuals to plot

n = 10


# ============================================================
# 2. COMMAND-LINE ARGUMENT
# ============================================================

parser = argparse.ArgumentParser(
    description="Plot CDMetaPOP pedigree trees."
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing pedigree_tree_output.csv"
)

args = parser.parse_args()

output_dir = Path(args.d)


# ============================================================
# 3. READ PEDIGREE
# ============================================================

pedigree_file = (
    output_dir /
    "pedigree_tree_output.csv"
)


if not pedigree_file.exists():

    raise FileNotFoundError(
        f"Could not find {pedigree_file}\n"
        f"Run tracking_pedigreetree_ai.py first."
    )


print(
    f"Reading pedigree: {pedigree_file}"
)


pedigree = pd.read_csv(
    pedigree_file
)


# ============================================================
# 4. FIND FINAL GENERATION
# ============================================================

last_generation = pedigree[
    "generation"
].max()


final_generation = pedigree[
    pedigree["generation"] == last_generation
].copy()


print(
    f"\nFinal generation: "
    f"{last_generation}"
)

print(
    f"Individuals in final generation: "
    f"{len(final_generation)}"
)


# ============================================================
# 5. SELECT INDIVIDUALS
# ============================================================

if n > len(final_generation):

    print(
        f"\nWARNING: n={n}, but only "
        f"{len(final_generation)} individuals "
        f"exist in the final generation."
    )

    n = len(final_generation)


# Select the first n individuals.
#
# We can later make this random or allow the user
# to specify particular individuals.

selected = final_generation.head(n)


print(
    f"\nPlotting {len(selected)} individuals:"
)

for tracking_id in selected[
    "tracking_id"
]:

    print(
        " ",
        tracking_id
    )


# ============================================================
# 6. CREATE UNIQUE INDIVIDUAL LOOKUP
# ============================================================

# Each individual may appear in multiple generations in
# pedigree_tree_output.csv because the same individual is
# recorded whenever it is present in an ind*.csv file.
#
# For constructing the pedigree, however, we only need ONE
# record per individual.
#
# Before removing duplicates, check that an individual's
# recorded parents are consistent.

parent_consistency_problems = []


for tracking_id, group in pedigree.groupby(
    "tracking_id"
):

    mother_ids = (
        group["mother_tracking_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    father_ids = (
        group["father_tracking_id"]
        .dropna()
        .astype(str)
        .unique()
    )


    if len(mother_ids) > 1:

        parent_consistency_problems.append({
            "tracking_id": tracking_id,
            "problem": "MOTHER ID CHANGED",
            "values": list(mother_ids)
        })


    if len(father_ids) > 1:

        parent_consistency_problems.append({
            "tracking_id": tracking_id,
            "problem": "FATHER ID CHANGED",
            "values": list(father_ids)
        })


if len(parent_consistency_problems) > 0:

    print("\nWARNING: inconsistent parentage found!")

    for problem in parent_consistency_problems:

        print(
            problem["tracking_id"],
            problem["problem"],
            problem["values"]
        )


# ------------------------------------------------------------
# Keep one row per individual
# ------------------------------------------------------------

individual_lookup = (
    pedigree
    .drop_duplicates(
        "tracking_id"
    )
    .set_index(
        "tracking_id"
    )
)


## ============================================================
# 7. FUNCTION TO FIND ALL ANCESTORS
# ============================================================

def find_ancestors(
    tracking_id,
    lookup,
    graph,
    visited=None
):

    """
    Recursively walk backwards through the pedigree.

    Parent → child edges are added to the graph.

    Each individual is only processed once.
    """

    if visited is None:

        visited = set()


    # Prevent circular ancestry from causing
    # infinite recursion.

    if tracking_id in visited:

        return


    visited.add(
        tracking_id
    )


    # --------------------------------------------------------
    # Individual isn't present in the supplied pedigree
    # --------------------------------------------------------

    if tracking_id not in lookup.index:

        return


    row = lookup.loc[
        tracking_id
    ]


    # --------------------------------------------------------
    # Mother
    # --------------------------------------------------------

    mother = row[
        "mother_tracking_id"
    ]


    if pd.notna(mother):

        mother = str(mother)


        if mother not in [
            "-1",
            "nan",
            "None"
        ] and mother != tracking_id:

            graph.add_edge(
                mother,
                tracking_id,
                parent="mother"
            )


            find_ancestors(
                mother,
                lookup,
                graph,
                visited
            )


    # --------------------------------------------------------
    # Father
    # --------------------------------------------------------

    father = row[
        "father_tracking_id"
    ]


    if pd.notna(father):

        father = str(father)


        if father not in [
            "-1",
            "nan",
            "None"
        ] and father != tracking_id:

            graph.add_edge(
                father,
                tracking_id,
                parent="father"
            )


            find_ancestors(
                father,
                lookup,
                graph,
                visited
            )


# ============================================================
# 8. FUNCTION TO CREATE A GENERATION LAYOUT
# ============================================================

def create_layout(
    graph,
    lookup
):

    """
    Arrange the pedigree vertically according to generation.

    Older generations are placed higher in the figure.
    """

    positions = {}

    generation_nodes = {}


    # --------------------------------------------------------
    # Determine generation of each node
    # --------------------------------------------------------

    for node in graph.nodes:

        if node in lookup.index:

            generation = int(
                lookup.loc[
                    node,
                    "generation"
                ]
            )

        else:

            # Parent not found in supplied data.
            #
            # Put it one generation above
            # the oldest known descendant.

            generation = -1


        if generation not in generation_nodes:

            generation_nodes[
                generation
            ] = []


        generation_nodes[
            generation
        ].append(node)


    # --------------------------------------------------------
    # Position nodes
    # --------------------------------------------------------

    for generation in sorted(
        generation_nodes
    ):

        nodes = sorted(
            generation_nodes[generation]
        )

        number = len(nodes)


        for i, node in enumerate(nodes):

            if number == 1:

                x = 0

            else:

                x = (
                    i -
                    (number - 1) / 2
                )


            positions[node] = (
                x,
                generation
            )


    return positions


# ============================================================
# 9. PLOT EACH INDIVIDUAL
# ============================================================

plot_directory = (
    output_dir /
    "pedigree_plots"
)


plot_directory.mkdir(
    exist_ok=True
)


for index, (_, individual) in enumerate(
    selected.iterrows(),
    start=1
):

    target = individual[
        "tracking_id"
    ]


    print(
        f"\nCreating tree {index}/{len(selected)}: "
        f"{target}"
    )


    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    G = nx.DiGraph()


    G.add_node(
        target
    )


    # Add all ancestors

    find_ancestors(
        target,
        individual_lookup,
        G
    )


    # --------------------------------------------------------
    # Add node metadata
    # --------------------------------------------------------

    for node in G.nodes:
        if node in individual_lookup.index:

            row = individual_lookup.loc[
                node
            ]

            G.nodes[node][
                "generation"
            ] = row["generation"]

            if "PatchID" in pedigree.columns:

                G.nodes[node][
                    "PatchID"
                ] = row["PatchID"]

            if "sex" in pedigree.columns:

                G.nodes[node][
                    "sex"
                ] = row["sex"]


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    pos = create_layout(
        G,
        individual_lookup
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )


    # --------------------------------------------------------
    # Draw edges
    # --------------------------------------------------------

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=12,
        width=1.0,
        alpha=0.5
    )


    # --------------------------------------------------------
    # Draw nodes
    # --------------------------------------------------------

    node_colors = []

    for node in G.nodes:

        sex = G.nodes[node].get("sex")

        if sex == "MXY":
            node_colors.append("blue")

        elif sex == "FXX":
            node_colors.append("red")

        else:
            node_colors.append("grey")


    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=250,
        node_color=node_colors,
        alpha=0.9
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=7
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        f"Pedigree of {target}"
    )


    ax.set_ylabel(
        "Generation"
    )


    ax.set_yticks(
        sorted(
            set(
                y
                for x, y in pos.values()
            )
        )
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"pedigree_{index}_"
        f"{target.replace('/', '_')}.png"
    )


    output_file = (
        plot_directory /
        filename
    )


    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Created {len(selected)} pedigree plots."
)

print(
    f"Plots saved in:"
)

print(
    plot_directory
)
