import pandas as pd
from pathlib import Path
import re


# ============================================================
# 1. SETTINGS
# ============================================================
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Track individuals through generations.")
parser.add_argument("-d", type=str, default="", help="Directory containign csv files")
args = parser.parse_args()
f = args.d
output_dir = Path(f)


# ============================================================
# 2. FIND THE INDIVIDUAL FILES
# ============================================================

ind_files = []

for file in output_dir.glob("ind*.csv"):

    match = re.fullmatch(r"ind(\d+)\.csv", file.name)

    if match:
        generation = int(match.group(1))
        ind_files.append((generation, file))

# Sort by generation number
ind_files.sort(key=lambda x: x[0])


print("Files being analysed:")

for generation, file in ind_files:
    print(generation, file.name)


# ============================================================
# 3. READ THE FIRST GENERATION
# ============================================================

first_generation, first_file = ind_files[0]

first = pd.read_csv(first_file)

print("\nFirst generation:")
print(first_file)

print("\nColumns:")
print(first.columns.tolist())


# ============================================================
# 4. CREATE THE UNIQUE INDIVIDUAL ID
# ============================================================

def get_last_three(id_string):

    parts = str(id_string).split("_")

    return "_".join(parts[-3:])


first["tracking_id"] = first["ID"].apply(get_last_three)


# ============================================================
# 5. CHECK THAT THE TRACKING ID IS UNIQUE
# ============================================================

duplicates = first[
    first.duplicated("tracking_id", keep=False)
]

if len(duplicates) > 0:

    print("\nWARNING: duplicate tracking IDs!")

    print(
        duplicates[
            ["ID", "tracking_id"]
        ].sort_values("tracking_id")
    )

else:

    print(
        "\nAll last-three-field IDs are unique "
        "in the first generation."
    )


# ============================================================
# 6. STORE THE FIRST-GENERATION INDIVIDUALS
# ============================================================

initial_individuals = set(
    first["tracking_id"]
)


# ============================================================
# 7. FOLLOW EVERY INDIVIDUAL THROUGH EVERY GENERATION
# ============================================================

records = []


for generation, file in ind_files:

    print(
        f"Reading generation {generation}: "
        f"{file.name}"
    )

    df = pd.read_csv(file)

    # Construct our proposed permanent individual ID
    df["tracking_id"] = df["ID"].apply(
        get_last_three
    )

    # Keep only individuals that existed
    # in the first generation
    df = df[
        df["tracking_id"].isin(
            initial_individuals
        )
    ]

    # Store the PatchID for each individual
    for _, row in df.iterrows():

        record = {
            "generation": generation,
            "tracking_id": row["tracking_id"],
            "ID": row["ID"],
            "PatchID": row["PatchID"]
        }

        records.append(record)


# ============================================================
# 8. CONVERT THE HISTORY INTO A DATAFRAME
# ============================================================

history = pd.DataFrame(records)


# ============================================================
# 9. SORT THE RESULTS
# ============================================================

history = history.sort_values(
    ["tracking_id", "generation"]
)


# ============================================================
# 10. PRINT A SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Individuals in first generation: "
    f"{len(initial_individuals)}"
)

print(
    f"Individuals subsequently observed: "
    f"{history['tracking_id'].nunique()}"
)

print(
    f"Total individual-generation observations: "
    f"{len(history)}"
)


# ============================================================
# 11. DISPLAY EXAMPLE TRAJECTORIES
# ============================================================

print("\nExample individual trajectories:\n")

for tracking_id, group in history.groupby(
    "tracking_id"
):

    print(
        tracking_id,
        ":",
        list(
            zip(
                group["generation"],
                group["PatchID"]
            )
        )
    )


    for i in range(1, len(group)):

        previous_patch = group["PatchID"].iloc[i - 1]
        current_patch = group["PatchID"].iloc[i]

        if previous_patch != current_patch:
            print(
                f"  Moved from {previous_patch} "
                f"to {current_patch}"
            )


    """
    # Only show the first 10 individuals
    if list(
        history["tracking_id"].unique()
    ).index(tracking_id) >= 9:
        break
    """