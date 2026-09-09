import numpy as np
from RunFile_Params import (
    n,
    ProbDist,
    param1,
    SaveDirName,
    repeats,
    batch_size,
)

import Landscape_Construction
import Matrix_Analysis
import Heterozygosity_ai

import time
import subprocess
import os
import shutil
import argparse


starttime = time.time()


#=============================================================================#
# ARGPARSER
#=============================================================================#

parser = argparse.ArgumentParser(
    prog="RGG_Construction.py",
    description="Create landscapes and run CDMetaPOP in batches"
)

parser.add_argument(
    "-i",
    type=str,
    default="default_inputs/inputs",
    help="inputs directory for simulation"
)

args = parser.parse_args()


#=============================================================================#
# SAVING DETAILS
#=============================================================================#

os.makedirs(SaveDirName, exist_ok=True)

try:
    shutil.copy("RunFile_Params.py", SaveDirName)
except FileNotFoundError:
    print("RunFile_Params.py not found, not copied to SaveDirName")


#=============================================================================#
# CREATE LANDSCAPES
#=============================================================================#

print("Starting to create landscapes")

repeat_dirs = []

for rep in range(repeats):

    print(f"Creating landscape for repeat {rep}")

    directoryname = os.path.join(
        SaveDirName,
        f"Repeat_{rep}"
    )

    repeat_dirs.append((rep, directoryname))

    Landscape_Construction.main(
        n=n,
        ProbDist=ProbDist,
        param1=param1,
        d=directoryname,
        i=args.i,
        r=rep
    )

print("Finished creating landscapes")


#=============================================================================#
# SPLIT INTO BATCHES
#=============================================================================#

Repeat_Matrix = [
    repeat_dirs[i:i + batch_size]
    for i in range(0, len(repeat_dirs), batch_size)
]

print(
    f"\nRunning {repeats} repeats in "
    f"{len(Repeat_Matrix)} batches "
    f"of maximum size {batch_size}"
)


#=============================================================================#
# STORAGE FOR ANALYSIS RESULTS
#=============================================================================#

Exponential_Decay_parameters = []
matrixlist = []


#=============================================================================#
# RUN SIMULATIONS BATCH BY BATCH
#=============================================================================#

for batch_number, batch in enumerate(Repeat_Matrix):

    print("\n" + "#" * 70)
    print(
        f"BATCH {batch_number + 1} / {len(Repeat_Matrix)}"
    )
    print("#" * 70)


    #=========================================================================#
    # RUN CDMetaPOP
    #=========================================================================#

    print("\nRunning CDMetaPOP simulations")

    plist = []

    for repeat_number, repeat_dir in batch:

        print(f"Starting Repeat_{repeat_number}")

        p = subprocess.Popen([
            "nice",
            "-n",
            "19",
            "uv",
            "run",
            "../../../src/CDmetaPOP.py",
            os.path.join(repeat_dir, "inputs"),
            "RunVars.csv",
            "../outputs/raw/"
        ])

        plist.append((repeat_number, p))


    # Wait for all simulations in this batch
    for repeat_number, p in plist:

        returncode = p.wait()

        if returncode != 0:
            print(
                f"WARNING: CDMetaPOP for Repeat_{repeat_number} "
                f"finished with return code {returncode}"
            )


    print("Batch simulations finished")


    #=========================================================================#
    # ANALYSE EACH REPEAT
    #=========================================================================#

    for repeat_number, repeat_dir in batch:

        print("\n" + "=" * 70)
        print(f"Processing Repeat {repeat_number}")
        print("=" * 70)


        #=====================================================================#
        # MATRIX ANALYSIS
        #=====================================================================#

        cdmatrix_path = os.path.join(
            repeat_dir,
            "inputs",
            "cdmats",
            "cdmatrix.csv"
        )

        if not os.path.isfile(cdmatrix_path):

            print(
                f"WARNING: cdmatrix.csv does not exist "
                f"for Repeat_{repeat_number}: "
                f"{cdmatrix_path}"
            )

            continue


        print("CD matrix:", cdmatrix_path)

        matrix = np.genfromtxt(cdmatrix_path, delimiter=",")


        #=====================================================================#
        # FIND RAW SIMULATION DIRECTORY
        #=====================================================================#

        raw_dir = os.path.join(
            repeat_dir,
            "outputs",
            "raw"
        )

        if not os.path.isdir(raw_dir):

            print(
                f"WARNING: raw directory does not exist "
                f"for Repeat_{repeat_number}: "
                f"{raw_dir}"
            )

            continue


        raw_subdirs = [
            dirname
            for dirname in os.listdir(raw_dir)
            if os.path.isdir(
                os.path.join(raw_dir, dirname)
            )
        ]


        if len(raw_subdirs) == 0:

            print(
                f"WARNING: No directory found inside raw "
                f"for Repeat_{repeat_number}"
            )

            continue


        if len(raw_subdirs) > 1:

            print(
                f"WARNING: Multiple directories found inside raw "
                f"for Repeat_{repeat_number}: "
                f"{raw_subdirs}"
            )

            print("Skipping this repeat.")

            continue


        analysis_dir = os.path.join(
            raw_dir,
            raw_subdirs[0]
        )

        print("Analysis directory:", analysis_dir)


        #=====================================================================#
        # HETEROZYGOSITY ANALYSIS
        #=====================================================================#

        try:

            decay_parameter = Heterozygosity_ai.main(
                analysis_dir,
                True,
                repeat_number
            )

            # Only save the results once the heterozygosity
            # calculation has successfully completed

            Exponential_Decay_parameters.append(
                decay_parameter
            )

            matrixlist.append(
                matrix
            )


        except Exception as e:

            print(
                f"ERROR analysing Repeat_{repeat_number}: {e}"
            )

            print(
                "Raw data will NOT be deleted so that "
                "the error can be investigated."
            )

            continue


        #=====================================================================#
        # DELETE RAW DATA
        #=====================================================================#

        print(
            f"Analysis complete. "
            f"Deleting raw data for Repeat_{repeat_number}"
        )

        shutil.rmtree(analysis_dir)

        print(
            f"Raw data deleted for Repeat_{repeat_number}"
        )


    print(
        f"\nFinished batch "
        f"{batch_number + 1} / {len(Repeat_Matrix)}"
    )

#=============================================================================#
# SAVE ANALYSIS RESULTS
#=============================================================================#

results_path = os.path.join(
    SaveDirName,
    "Analysis_Results.npz"
)

np.savez(
    results_path,
    matrixlist=np.asarray(matrixlist, dtype=object),
    Exponential_Decay_parameters=np.asarray(
        Exponential_Decay_parameters,
        dtype=object
    )
)

print(f"\nSaved analysis results to:")
print(results_path)
#=============================================================================#
# FINISHED
#=============================================================================#

print("\nFinished all landscape simulations and analyses")

endtime = time.time()

print(
    f"Total time taken: "
    f"{endtime - starttime:.2f} seconds"
)

"""
import numpy as np
from RunFile_Params import n, ProbDist, param1,SaveDirName,repeats,batch_num,batch_size
import Landscape_Construction

import time

import subprocess
import os,shutil
import argparse
import copy

starttime = time.time()

#=============================================================================#
# ARGPARSER
#=============================================================================#
parser = argparse.ArgumentParser(
    prog = "RGG_Construction.py",
    description="Create a landscape for CDmetaPOP")



parser.add_argument("-i",
                    type=str,
                    default='default_inputs/inputs',
                    help='inputs directory for simulation')


args = parser.parse_args()





#=============================================================================#
# Saving Details
#=============================================================================#
if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Directory")

try:
    shutil.copy("RunFile_Params.py",SaveDirName)
except:
    print("RunFile_Params.py not found, not copied to SaveDirName")



print("Starting to create landscapes")
#Create the landscapes
Repeat_Matrix = []
Repeat_Directories = []
for rep in range(repeats):
    print(f"Creating landscape for repeat {rep}")
    directoryname = SaveDirName+"/Repeat_%d"%rep   
    Repeat_Directories.append(directoryname)
    Landscape_Construction.main(n=n, ProbDist=ProbDist, param1=param1, d=directoryname, i=args.i,r=rep)

    if len(Repeat_Directories) == batch_size:
        Repeat_Matrix.append(copy.copy(Repeat_Directories))
        Repeat_Directories = []

print("Finished creating landscapes")



print("Starting landscape simulations")
#Execute the landscape simulations in this format:
# uv run ../../src/CDmetaPOP.py RGG_n20_ProbDist_Power_param1_4.0_nonperiodic_seed0/inputs RunVars.csv ../outputs/raw/


for Repeat_Directories in Repeat_Matrix:
    #RUN SIMULATIONS
    plist = []
    for i in Repeat_Directories:
        p=subprocess.Popen(['nice','-n','19','uv','run','../../../src/CDmetaPOP.py',str(i)+'/inputs', 'RunVars.csv', '../outputs/raw/'])
        plist.append(p)

    for p in plist:
        p.wait()

    #EXTRACT HETEROZYGOSITY DECAY AND CDMATRIX
    plist = []
    for i in Repeat_Directories:
        p=subprocess.Popen(['nice','-n','19','uv','run','Extract_Heterozygosity_Decay.py',str(i)+'/outputs/raw/'])
        plist.append(p)

    for p in plist:
        p.wait()

    #Delete all CSVs in the outputs/raw directory to save space
    for i in Repeat_Directories:
        for file in os.listdir(str(i)+'/outputs/raw/'):
            if file.endswith(".csv"):
                os.remove(str(i)+'/outputs/raw/'+file)





print("Finished landscape simulations")


endtime = time.time()
print(f"Total time taken: {endtime - starttime} seconds")
"""