import numpy as np
import matplotlib.pyplot as plt

import Heterozygosity_ai
import Matrix_Analysis

import time

import subprocess
import os, shutil
import argparse
import re

starttime = time.time()

#=============================================================================#
# ARGPARSER
#=============================================================================#
parser = argparse.ArgumentParser(
    prog="Heterozygosity_Run.py",
    description="Run heterozygosity analysis over all landscape repeats"
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing all the landscape repeats"
)

args = parser.parse_args()


#=============================================================================#
# FIND REPEAT DIRECTORIES
#=============================================================================#

SaveDir = args.d

repeat_dirs = []

for dirname in os.listdir(SaveDir):

    full_path = os.path.join(SaveDir, dirname)

    if not os.path.isdir(full_path):
        continue

    match = re.fullmatch(r"Repeat_(\d+)", dirname)

    if match:
        repeat_number = int(match.group(1))
        repeat_dirs.append((repeat_number, full_path))

repeat_dirs.sort(key=lambda x: x[0])

print("Found", len(repeat_dirs), "repeats")


#=============================================================================#
# ANALYSE EACH REPEAT
#=============================================================================#

Exponential_Decay_parameters = []
statslist = []

for repeat_number, repeat_dir in repeat_dirs:

    print("\n" + "=" * 70)
    print("Processing Repeat", repeat_number)
    print("=" * 70)


    #========================================================================#
    # FIND CDMATRIX FOR THIS REPEAT
    #========================================================================#

    cdmatrix_path = os.path.join(
        repeat_dir,
        "inputs",
        "cdmats",
        "cdmatrix.csv"
    )

    if not os.path.isfile(cdmatrix_path):

        print(
            f"WARNING: cdmatrix.csv does not exist for "
            f"Repeat_{repeat_number}: {cdmatrix_path}"
        )

        continue

    print("CD matrix:")
    print(cdmatrix_path)

    # Placeholder for future matrix analysis
    statslist.append(Matrix_Analysis.main(cdmatrix_path))


    #========================================================================#
    # FIND RANDOM-NUMBER DIRECTORY FOR HETEROZYGOSITY ANALYSIS
    #========================================================================#

    raw_dir = os.path.join(
        repeat_dir,
        "outputs",
        "raw"
    )

    if not os.path.isdir(raw_dir):

        print(
            f"WARNING: raw directory does not exist for "
            f"Repeat_{repeat_number}: {raw_dir}"
        )

        continue

    raw_subdirs = [
        dirname
        for dirname in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, dirname))
    ]

    if len(raw_subdirs) == 0:

        print(
            f"WARNING: No directory found inside raw for "
            f"Repeat_{repeat_number}"
        )

        continue

    if len(raw_subdirs) > 1:

        print(
            f"WARNING: Multiple directories found inside raw for "
            f"Repeat_{repeat_number}: {raw_subdirs}"
        )

        print("Skipping this repeat.")

        continue

    random_dir = raw_subdirs[0]

    analysis_dir = os.path.join(
        raw_dir,
        random_dir
    )

    print("Analysis directory:")
    print(analysis_dir)


    #========================================================================#
    # RUN HETEROZYGOSITY ANALYSIS
    #========================================================================#

    Exponential_Decay_parameters.append(
        Heterozygosity_ai.main(analysis_dir)
    )


#=============================================================================#
# FINISHED
#============================================================================#

print(Exponential_Decay_parameters)

endtime = time.time()

print("\n" + "=" * 70)
print("All repeats processed")
print("Time taken:", endtime - starttime, "seconds")
print("=" * 70)



#Plotting
Mixing_Times = [x["mixing_time"] for x in statslist]

plt.loglog(Mixing_Times, Exponential_Decay_parameters, 'o')
plt.xlabel("Mixing Time")
plt.ylabel("Heterozygosity Decay Parameter (a)")

plt.savefig(str(SaveDir) + "/Exponential_Decay_vs_Mixing_Time.png")
plt.show()