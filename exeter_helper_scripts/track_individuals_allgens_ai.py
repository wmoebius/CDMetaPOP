import pandas as pd
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt

# ============================================================
# 1. SETTINGS
# ============================================================

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Track individuals through generations."
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
# 2. FIND THE INDIVIDUAL FILES
# ============================================================

ind_files = []

for file in output_dir.glob("ind*.csv"):

    match = re.fullmatch(
        r"ind(\d+)\.csv",
        file.name
    )

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


first["tracking_id"] = first["ID"].apply(
    get_last_three
)


# ============================================================
# 5. CHECK THAT THE TRACKING ID IS UNIQUE
# ============================================================

duplicates = first[
    first.duplicated(
        "tracking_id",
        keep=False
    )
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
# 6. FOLLOW EVERY INDIVIDUAL THROUGH EVERY GENERATION
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

    # Store EVERY individual in this generation.
    #
    # We deliberately do NOT filter using the
    # first-generation individuals. This means individuals
    # born in later generations are included as well.

    for _, row in df.iterrows():

        record = {
            "generation": generation,
            "tracking_id": row["tracking_id"],
            "ID": row["ID"],
            "PatchID": row["PatchID"]
        }

        records.append(record)


# ============================================================
# 7. CONVERT THE HISTORY INTO A DATAFRAME
# ============================================================

history = pd.DataFrame(records)


# ============================================================
# 8. SORT THE RESULTS
# ============================================================

history = history.sort_values(
    ["tracking_id", "generation"]
)


# ============================================================
# 9. PRINT A SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Total unique individuals: "
    f"{history['tracking_id'].nunique()}"
)

print(
    f"Total individual-generation observations: "
    f"{len(history)}"
)


# ============================================================
# 10. DISPLAY EXAMPLE TRAJECTORIES
# ============================================================

print("\nExample individual trajectories:\n")


#Histogram list to store the number of years an individual lives for.
#Not very rigorous, as doeeeeeeeeeeeeeeeeeeesn't account for those still alive at the end or the instantiation of agesat the start.
yearHist = []

for i, (tracking_id, group) in enumerate(
    history.groupby("tracking_id")
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

    #Histogram of lifetimes populated
    yearHist.append(len(list(zip(group["generation"], group["PatchID"]))))


    for j in range(1, len(group)):

        previous_patch = group["PatchID"].iloc[j - 1]
        current_patch = group["PatchID"].iloc[j]

        if previous_patch != current_patch:

            print(
                f"  Moved from {previous_patch} "
                f"to {current_patch}"
            )

    # Only show the first 500 individuals
    #if i >= 500:
    #    break


plt.hist(yearHist, bins=range(1, max(yearHist) + 2), align='left', edgecolor='black',log=True)
plt.xlabel('Number of Years Lived')
plt.ylabel('Number of Individuals')
plt.title('Histogram of Individual Lifetimes')
plt.xticks(range(1, max(yearHist) + 1))
plt.show()