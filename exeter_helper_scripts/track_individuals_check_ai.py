"""
ai generated code to check that the last three elements of the id corespond to the unique label for an individual.
To check, we follow the individuals of the first generation and if there are any changes to gender or the age misbehaves, then we have a problem and the last three
element are not unique.
"""


import pandas as pd
from pathlib import Path
import re
import argparse

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

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
# Only match:
#   ind1.csv
#   ind2.csv
#   ind10.csv
# etc.
#
# This deliberately excludes:
#   indSample1.csv
#   indSample2.csv
#   etc.

ind_files = []

for file in output_dir.glob("ind*.csv"):

    match = re.fullmatch(r"ind(\d+)\.csv", file.name)

    if match:
        generation = int(match.group(1))
        ind_files.append((generation, file))

# Sort numerically by generation
ind_files.sort(key=lambda x: x[0])

# Just the Path objects, in chronological order
ind_files = [file for generation, file in ind_files]

print("Files being analysed:")

for file in ind_files:
    print(file.name)


a = input("Press enter to continue of crtl-c to abort.")

# --------------------------------------------------
# READ FIRST GENERATION
# --------------------------------------------------

first_file = ind_files[0]
first = pd.read_csv(first_file)

print(f"First generation file: {first_file}")
print(f"Number of individuals: {len(first)}")
print(first.columns.tolist())


# --------------------------------------------------
# EXTRACT LAST THREE ID FIELDS
# --------------------------------------------------

def get_last_three(id_string):
    parts = str(id_string).split("_")
    return "_".join(parts[-3:])


first["tracking_id"] = first["ID"].apply(get_last_three)

# Check whether the proposed ID is actually unique
duplicates = first[first.duplicated("tracking_id", keep=False)]

if len(duplicates) > 0:
    print("\nWARNING: duplicate last-three-field IDs in first generation!")
    print(duplicates[["ID", "tracking_id"]].sort_values("tracking_id"))
else:
    print("\nAll last-three-field IDs are unique in the first generation.")


# --------------------------------------------------
# IDENTIFY VARIABLES WE WANT TO TRACK
# --------------------------------------------------

print("\nColumns available:")
for i, col in enumerate(first.columns):
    print(i, col)


# Change these names if necessary after looking at the output
TRACK_COLUMNS = [
    "age",
    "sex"
]


# --------------------------------------------------
# STORE FIRST-GENERATION STATE
# --------------------------------------------------

initial_state = first.set_index("tracking_id")[TRACK_COLUMNS].copy()


# --------------------------------------------------
# FOLLOW EACH INDIVIDUAL
# --------------------------------------------------

records = []

for file in ind_files:
    print("Checking file:", file.name)
    df = pd.read_csv(file)

    df["tracking_id"] = df["ID"].apply(get_last_three)

    # Keep only individuals that existed in generation 1
    df = df[df["tracking_id"].isin(initial_state.index)]

    for _, row in df.iterrows():

        record = {
            "file": file.name,
            "tracking_id": row["tracking_id"],
            "ID": row["ID"],
        }

        for col in TRACK_COLUMNS:
            record[col] = row[col]

        records.append(record)


history = pd.DataFrame(records)


# --------------------------------------------------
# CHECK FOR AGE / SEX CHANGES
# --------------------------------------------------

print("\nChecking individuals...\n")

problems = []

for tracking_id, group in history.groupby("tracking_id"):

    group = group.copy()

    # Sort chronologically according to file order
    group["file_order"] = group["file"].apply(
        lambda x: ind_files.index(output_dir / x)
    )

    group = group.sort_values("file_order")

    initial = group.iloc[0]

    # -------------------------
    # Sex should never change
    # -------------------------

    if "sex" in TRACK_COLUMNS:

        sexes = group["sex"].dropna().unique()

        if len(sexes) > 1:
            problems.append({
                "tracking_id": tracking_id,
                "problem": "SEX CHANGED",
                "values": list(sexes),
                "history": group[["file", "ID", "sex"]].to_dict("records")
            })

    # -------------------------
    # Age should increase
    # -------------------------

    if "age" in TRACK_COLUMNS:

        ages = group["age"].dropna().tolist()

        for i in range(1, len(ages)):

            if ages[i] != ages[i-1] + 1:

                problems.append({
                    "tracking_id": tracking_id,
                    "problem": "AGE DID NOT INCREASE BY 1",
                    "values": ages,
                    "history": group[["file", "ID", "age"]].to_dict("records")
                })

                break



 
# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Individuals in first generation: {len(initial_state)}")
print(f"Individuals subsequently observed: {history['tracking_id'].nunique()}")
print(f"Problems found: {len(problems)}")

if len(problems) == 0:

    print("\nNo age/sex inconsistencies found.")
    print("This supports the hypothesis that the last three ID fields")
    print("continue to identify the same individual.")

else:

    print("\nProblems found:\n")

    for p in problems:

        print("-" * 60)
        print("Tracking ID:", p["tracking_id"])
        print("Problem:", p["problem"])
        print("Values:", p["values"])

        print("\nHistory:")
        for h in p["history"]:
            print(h)
