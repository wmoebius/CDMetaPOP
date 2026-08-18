import pandas as pd
from pathlib import Path
import re
import argparse


parser = argparse.ArgumentParser(
    description = "Create a layer of histograms for the ancestry of a sub-population.")

parser.add_argument(
    "-d",
    type= str,
    required = True,
    help='Directory containing ind<number>.csv files'
)

parser.add_argument(
    "-g",
    type = str,
    required = True,
    help = "The genotype in question"
)

parser.add_argument(
    "-PID",
    type = str,
    required = True,
    help = "The patch ID"
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
# FUNCTION TO EXTRACT LAST THREE ID FIELDS
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
# Extract the last three ID fields, mother and Father ID's from the last ind file, for the selected genotype and patchID
# ============================================================

