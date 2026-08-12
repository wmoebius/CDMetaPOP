"""
Construct a complete CDMetaPOP pedigree table.

Usage:

    uv run tracking_pedigreetree_ai.py -d outputdirectory

The script:

1. Finds ind<number>.csv files.
2. Ignores indSample*.csv and other non-matching files.
3. Extracts the last three fields of ID as the permanent
   individual identifier.
4. Extracts the last three fields of MID and FID as the
   permanent mother/father identifiers.
5. Records generation, patch, coordinates, sex, age, etc.
6. Writes the complete pedigree to:

       pedigree_tree_output.csv
"""


import pandas as pd
from pathlib import Path
import re
import argparse


# ============================================================
# 1. SETTINGS
# ============================================================

parser = argparse.ArgumentParser(
    description="Construct a CDMetaPOP pedigree table."
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
# 3. FIND INDIVIDUAL FILES
# ============================================================

ind_files = []

for file in output_dir.glob("ind*.csv"):

    # Matches:
    #
    # ind0.csv
    # ind1.csv
    # ind2.csv
    # ind10.csv
    #
    # Does NOT match:
    #
    # indSample1.csv
    # indSample2.csv

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


# Sort numerically

ind_files.sort(
    key=lambda x: x[0]
)


if len(ind_files) == 0:

    raise FileNotFoundError(
        f"No ind<number>.csv files found in "
        f"{output_dir}"
    )


print("=" * 70)
print("CDMetaPOP PEDIGREE EXTRACTION")
print("=" * 70)

print("\nFiles being analysed:")

for generation, file in ind_files:

    print(
        f"Generation {generation}: "
        f"{file.name}"
    )


# ============================================================
# 4. FUNCTION TO EXTRACT LAST THREE ID FIELDS
# ============================================================

def get_last_three(id_string):

    """
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

records = []


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
    # Create permanent IDs
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
    # Store individuals
    # --------------------------------------------------------

    for _, row in df.iterrows():

        record = {

            "generation": generation,

            "tracking_id":
                row["tracking_id"],

            "ID":
                row["ID"],

            "mother_tracking_id":
                row["mother_tracking_id"],

            "father_tracking_id":
                row["father_tracking_id"],

            "MID":
                row["MID"],

            "FID":
                row["FID"],
        }


        # ----------------------------------------------------
        # Include useful spatial / biological information
        # ----------------------------------------------------

        optional_columns = [

            "PatchID",

            "XCOORD",

            "YCOORD",

            "sex",

            "age",

            "size",

            "mature",

            "newmature",

            "layeggs",

            "state",

            "Species",

            "ClassFile",

            "SubPatchID"
        ]


        for column in optional_columns:

            if column in df.columns:

                record[column] = row[column]


        records.append(record)


# ============================================================
# 6. CREATE PEDIGREE DATAFRAME
# ============================================================

pedigree = pd.DataFrame(
    records
)


# ============================================================
# 7. SORT
# ============================================================

pedigree = pedigree.sort_values(
    [
        "generation",
        "tracking_id"
    ]
)


# ============================================================
# 8. BASIC SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Generations: "
    f"{pedigree['generation'].nunique()}"
)

print(
    f"Individual records: "
    f"{len(pedigree)}"
)

print(
    f"Unique tracking IDs: "
    f"{pedigree['tracking_id'].nunique()}"
)

print(
    f"First generation: "
    f"{pedigree['generation'].min()}"
)

print(
    f"Last generation: "
    f"{pedigree['generation'].max()}"
)


# ============================================================
# 9. CHECK FOR DUPLICATE TRACKING IDS
# ============================================================

duplicates = pedigree[
    pedigree.duplicated(
        [
            "generation",
            "tracking_id"
        ],
        keep=False
    )
]


if len(duplicates) > 0:

    print(
        "\nWARNING:"
    )

    print(
        "The following tracking IDs occur "
        "more than once within a generation:"
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


else:

    print(
        "\nNo duplicate tracking IDs "
        "within generations."
    )


# ============================================================
# 10. SAVE PEDIGREE
# ============================================================

output_file = (
    output_dir /
    "pedigree_tree_output.csv"
)


pedigree.to_csv(
    output_file,
    index=False
)


print(
    f"\nPedigree written to:"
)

print(
    output_file
)