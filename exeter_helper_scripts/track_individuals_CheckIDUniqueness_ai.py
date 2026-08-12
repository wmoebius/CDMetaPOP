"""
AI-generated code to check that the last three elements of the ID
correspond to a unique label for an individual.

The script scans EVERY individual in EVERY generation.

For each generation, it checks that the last three elements of the ID
are unique.

It then follows every proposed individual through all generations in
which they appear and checks whether characteristics that should remain
constant behave sensibly.

In particular:
    - sex should never change;
    - age should increase by one between consecutive generations.

If either occurs, this suggests that the last three elements of the ID
are NOT uniquely identifying an individual.
"""


import pandas as pd
from pathlib import Path
import re
import argparse


# ============================================================
# SETTINGS
# ============================================================

# Parse command-line arguments

parser = argparse.ArgumentParser(
    description="Check whether the last three ID fields uniquely identify individuals."
)

parser.add_argument(
    "-d",
    type=str,
    default="",
    help="Directory containing csv files"
)

args = parser.parse_args()

f = args.d
output_dir = Path(f)


# ============================================================
# FIND THE INDIVIDUAL FILES
# ============================================================

# Only match:
#
#     ind1.csv
#     ind2.csv
#     ind10.csv
#     ...
#
# This deliberately excludes:
#
#     indSample1.csv
#     indSample2.csv
#     ...

ind_files = []

for file in output_dir.glob("ind*.csv"):

    match = re.fullmatch(
        r"ind(\d+)\.csv",
        file.name
    )

    if match:

        generation = int(match.group(1))

        ind_files.append(
            (generation, file)
        )


# Sort numerically by generation

ind_files.sort(
    key=lambda x: x[0]
)


# Convert to just the Path objects

ind_files = [
    file
    for generation, file in ind_files
]


print("Files being analysed:")

for file in ind_files:
    print(file.name)


a = input(
    "Press enter to continue or Ctrl-C to abort."
)


# ============================================================
# READ FIRST GENERATION
# ============================================================

first_file = ind_files[0]

first = pd.read_csv(
    first_file
)

print(
    f"First generation file: {first_file}"
)

print(
    f"Number of individuals: {len(first)}"
)

print(
    first.columns.tolist()
)


# ============================================================
# EXTRACT LAST THREE ID FIELDS
# ============================================================

def get_last_three(id_string):

    parts = str(id_string).split("_")

    return "_".join(parts[-3:])


# ============================================================
# IDENTIFY VARIABLES WE WANT TO TRACK
# ============================================================

print("\nColumns available:")

for i, col in enumerate(first.columns):

    print(i, col)


# Change these names if necessary

TRACK_COLUMNS = [
    "age",
    "sex"
]


# ============================================================
# CHECK UNIQUENESS WITHIN EVERY GENERATION
# ============================================================

print("\n")
print("=" * 60)
print("CHECKING UNIQUENESS WITHIN EACH GENERATION")
print("=" * 60)


uniqueness_problems = []


for file in ind_files:

    print(
        f"Checking uniqueness in {file.name}"
    )

    df = pd.read_csv(file)

    # Construct the proposed individual ID

    df["tracking_id"] = df["ID"].apply(
        get_last_three
    )


    # Find duplicate tracking IDs

    duplicates = df[
        df.duplicated(
            "tracking_id",
            keep=False
        )
    ]


    if len(duplicates) > 0:

        print(
            f"\nWARNING: duplicate tracking IDs "
            f"found in {file.name}!"
        )

        print(
            duplicates[
                [
                    "ID",
                    "tracking_id"
                ]
            ].sort_values(
                "tracking_id"
            )
        )


        # Store the problem so it can also be
        # reported at the end

        for tracking_id, group in duplicates.groupby(
            "tracking_id"
        ):

            uniqueness_problems.append({

                "file": file.name,

                "tracking_id": tracking_id,

                "problem": (
                    "TRACKING ID NOT UNIQUE "
                    "WITHIN GENERATION"
                ),

                "IDs": group["ID"].tolist()
            })


# ============================================================
# FOLLOW EVERY INDIVIDUAL THROUGH EVERY GENERATION
# ============================================================

print("\n")
print("=" * 60)
print("BUILDING INDIVIDUAL HISTORIES")
print("=" * 60)


records = []


for file in ind_files:

    print(
        "Checking file:",
        file.name
    )


    df = pd.read_csv(file)


    # Create proposed permanent individual ID

    df["tracking_id"] = df["ID"].apply(
        get_last_three
    )


    # --------------------------------------------------------
    # Record EVERY individual
    # --------------------------------------------------------

    for _, row in df.iterrows():

        record = {

            "file": file.name,

            "tracking_id": row["tracking_id"],

            "ID": row["ID"],
        }


        # Add the variables we want to track

        for col in TRACK_COLUMNS:

            record[col] = row[col]


        records.append(record)


# Convert all records into one DataFrame

history = pd.DataFrame(
    records
)


# ============================================================
# SORT THE HISTORY
# ============================================================

history = history.sort_values(
    [
        "tracking_id",
        "file"
    ]
)


# ============================================================
# CHECK FOR AGE / SEX CHANGES
# ============================================================

print("\n")
print("=" * 60)
print("CHECKING INDIVIDUAL HISTORIES")
print("=" * 60)


problems = []


for tracking_id, group in history.groupby(
    "tracking_id"
):

    group = group.copy()


    # --------------------------------------------------------
    # Determine chronological order
    # --------------------------------------------------------

    group["file_order"] = group["file"].apply(
        lambda x: ind_files.index(
            output_dir / x
        )
    )


    group = group.sort_values(
        "file_order"
    )


    # --------------------------------------------------------
    # SEX SHOULD NEVER CHANGE
    # --------------------------------------------------------

    if "sex" in TRACK_COLUMNS:

        sexes = (
            group["sex"]
            .dropna()
            .unique()
        )


        if len(sexes) > 1:

            problems.append({

                "tracking_id": tracking_id,

                "problem": "SEX CHANGED",

                "values": list(sexes),

                "history": group[
                    [
                        "file",
                        "ID",
                        "sex"
                    ]
                ].to_dict(
                    "records"
                )
            })


    # --------------------------------------------------------
    # AGE SHOULD INCREASE BY ONE
    # --------------------------------------------------------

    if "age" in TRACK_COLUMNS:

        ages = (
            group["age"]
            .dropna()
            .tolist()
        )


        for i in range(
            1,
            len(ages)
        ):

            if ages[i] != ages[i - 1] + 1:

                problems.append({

                    "tracking_id": tracking_id,

                    "problem": (
                        "AGE DID NOT "
                        "INCREASE BY 1"
                    ),

                    "values": ages,

                    "history": group[
                        [
                            "file",
                            "ID",
                            "age"
                        ]
                    ].to_dict(
                        "records"
                    )
                })


                # Once we have found an age
                # problem for this individual,
                # there is no need to report
                # another age problem for them.

                break


# ============================================================
# REPORT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("RESULTS")
print("=" * 60)


print(
    f"Number of generations analysed: "
    f"{len(ind_files)}"
)


print(
    f"Total individual-generation observations: "
    f"{len(history)}"
)


print(
    f"Total unique tracking IDs: "
    f"{history['tracking_id'].nunique()}"
)


print(
    f"Tracking IDs duplicated within a generation: "
    f"{len(uniqueness_problems)}"
)


print(
    f"Individuals with age/sex problems: "
    f"{len(problems)}"
)


# ============================================================
# REPORT UNIQUENESS PROBLEMS
# ============================================================

if len(uniqueness_problems) > 0:

    print("\n")
    print("=" * 60)
    print("UNIQUENESS PROBLEMS")
    print("=" * 60)


    for p in uniqueness_problems:

        print("\n" + "-" * 60)

        print(
            "File:",
            p["file"]
        )

        print(
            "Tracking ID:",
            p["tracking_id"]
        )

        print(
            "Problem:",
            p["problem"]
        )

        print(
            "Actual IDs:"
        )

        for ID in p["IDs"]:

            print(
                "   ",
                ID
            )


# ============================================================
# REPORT AGE / SEX PROBLEMS
# ============================================================

if len(problems) > 0:

    print("\n")
    print("=" * 60)
    print("AGE / SEX PROBLEMS")
    print("=" * 60)


    for p in problems:

        print(
            "\n" + "-" * 60
        )

        print(
            "Tracking ID:",
            p["tracking_id"]
        )

        print(
            "Problem:",
            p["problem"]
        )

        print(
            "Values:",
            p["values"]
        )

        print(
            "\nHistory:"
        )


        for h in p["history"]:

            print(h)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n")
print("=" * 60)
print("CONCLUSION")
print("=" * 60)


if (
    len(uniqueness_problems) == 0
    and len(problems) == 0
):

    print(
        "\nNo inconsistencies were found."
    )

    print(
        "The last three ID fields were unique "
        "within every generation and produced "
        "biologically consistent individual "
        "histories across all generations."
    )

    print(
        "\nThis strongly supports the hypothesis "
        "that the last three ID fields uniquely "
        "identify individuals."
    )


else:

    print(
        "\nWARNING: inconsistencies were found."
    )

    print(
        "The last three ID fields may not uniquely "
        "identify individuals, or one of the "
        "assumptions about age/sex may be incorrect."
    )
