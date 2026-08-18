import pandas as pd
from pathlib import Path
import re
import argparse


parser = argparse.ArgumentParser(
    description="Read ind<number>.csv files into a list of DataFrames."
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


# ============================================================
# READ DATAFRAMES
# ============================================================

Generations = []

for generation, file in ind_files:

    df = pd.read_csv(file)

    Generations.append(df)


# ============================================================
# EXAMPLE
# ============================================================

print(
    f"Loaded {len(Generations)} generations."
)

for generation, df in enumerate(Generations):

    print(
        f"Generation {generation}: "
        f"{len(df)} individuals"
    )


def get_last_three(id_string):

    parts = str(id_string).split("_")

    return "_".join(parts[-3:])
#Simplify ID's 
for df in Generations:

    for column in ["ID", "MID", "FID"]:

        df[column] = df[column].apply(get_last_three)




Genotype_Individuals = []

for df in Generations:

    # Determine which locus/columns to use
    locus = args.g[:2]

    allele1 = int(args.g[3])
    allele2 = int(args.g[5])

    if locus == "L0":
        genotype_column_1 = "L0A0"
        genotype_column_2 = "L0A1"

    elif locus == "L1":
        genotype_column_1 = "L1A0"
        genotype_column_2 = "L1A1"

    else:
        raise ValueError(
            f"Invalid genotype: {args.g}"
        )

    # Select individuals matching patch and genotype
    matches = df[
        (df["PatchID"] == int(args.PID)) &
        (df[genotype_column_1] == allele1) &
        (df[genotype_column_2] == allele2)
    ]

    Genotype_Individuals.extend(
        matches["ID"].tolist()
    )

print(Genotype_Individuals)
"""
Location_List = [[0,0,0]*len(Generations_List)]
for GID in Genotype_Individuals:
    Buffer = [[GID,len(Generations_list)]]
    while len(Buffer) > 0:
        ID = Buffer[0][0]
        gen = Buffer[0][1]

        if ID in Generations_list[gen]:
            Location_List[gen][]
"""