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
import copy

from concurrent.futures import ProcessPoolExecutor, as_completed


#=============================================================================#
# WORKER FUNCTIONS
#=============================================================================#

def create_landscape(args):
    """
    Create one landscape.

    This function is run in a separate process.
    """

    rep, directoryname, input_dir = args

    print(f"Creating landscape for Repeat_{rep}")

    Landscape_Construction.main(
        n=n,
        ProbDist=ProbDist,
        param1=param1,
        d=directoryname,
        i=input_dir,
        r=rep
    )

    return rep, directoryname


def analyse_repeat(args):
    """
    Analyse one repeat and delete its raw data if analysis succeeds.

    This function is run in a separate process.
    """

    repeat_number, repeat_dir = args

    print("\n" + "=" * 70)
    print(f"Processing Repeat {repeat_number}")
    print("=" * 70)

    #=========================================================================#
    # MATRIX ANALYSIS
    #=========================================================================#

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

        return None

    print(
        f"Repeat_{repeat_number}: "
        f"CD matrix: {cdmatrix_path}"
    )

    matrix = np.genfromtxt(
        cdmatrix_path,
        delimiter=","
    )

    #=========================================================================#
    # FIND RAW SIMULATION DIRECTORY
    #=========================================================================#

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

        return None

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

        return None

    if len(raw_subdirs) > 1:

        print(
            f"WARNING: Multiple directories found inside raw "
            f"for Repeat_{repeat_number}: "
            f"{raw_subdirs}"
        )

        print("Skipping this repeat.")

        return None

    analysis_dir = os.path.join(
        raw_dir,
        raw_subdirs[0]
    )

    print(
        f"Repeat_{repeat_number}: "
        f"Analysis directory: {analysis_dir}"
    )

    #=========================================================================#
    # HETEROZYGOSITY ANALYSIS
    #=========================================================================#

    try:

        decay_parameter, heterozygosity_curves, population_curves = (
            Heterozygosity_ai.main(
                analysis_dir,
                True,
                True,
                True,
                repeat_number
            )
        )

    except Exception as e:

        print(
            f"ERROR analysing Repeat_{repeat_number}: {e}"
        )

        print(
            "Raw data will NOT be deleted so that "
            "the error can be investigated."
        )

        return None

    #=========================================================================#
    # DELETE RAW DATA
    #=========================================================================#

    try:

        print(
            f"Analysis complete. "
            f"Deleting raw data for Repeat_{repeat_number}"
        )

        shutil.rmtree(analysis_dir)

        print(
            f"Raw data deleted for Repeat_{repeat_number}"
        )

    except Exception as e:

        print(
            f"WARNING: Could not delete raw data for "
            f"Repeat_{repeat_number}: {e}"
        )

    #=========================================================================#
    # RETURN RESULTS
    #=========================================================================#

    return {
        "repeat_number": repeat_number,
        "matrix": matrix,
        "decay_parameter": decay_parameter,
        "heterozygosity_curves": copy.copy(
            heterozygosity_curves
        ),
        "population_curves": copy.copy(
            population_curves
        ),
    }


#=============================================================================#
# MAIN
#=============================================================================#

def main():

    starttime = time.time()

    #========================================================================#
    # ARGPARSER
    #========================================================================#

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

    #========================================================================#
    # SAVING DETAILS
    #========================================================================#

    os.makedirs(
        SaveDirName,
        exist_ok=True
    )

    try:

        shutil.copy(
            "RunFile_Params.py",
            SaveDirName
        )

    except FileNotFoundError:

        print(
            "RunFile_Params.py not found, "
            "not copied to SaveDirName"
        )

    #========================================================================#
    # CREATE REPEAT DIRECTORIES
    #========================================================================#

    repeat_dirs = []

    for rep in range(repeats):

        directoryname = os.path.join(
            SaveDirName,
            f"Repeat_{rep}"
        )

        repeat_dirs.append(
            (rep, directoryname)
        )

    #========================================================================#
    # SPLIT INTO BATCHES
    #========================================================================#

    Repeat_Matrix = [
        repeat_dirs[i:i + batch_size]
        for i in range(
            0,
            len(repeat_dirs),
            batch_size
        )
    ]

    print(
        f"\nRunning {repeats} repeats in "
        f"{len(Repeat_Matrix)} batches "
        f"of maximum size {batch_size}"
    )

    #========================================================================#
    # STORAGE FOR ANALYSIS RESULTS
    #========================================================================#

    Exponential_Decay_parameters = []
    heterozygosity_curves_list = []
    population_curves_list = []
    matrixlist = []

    #========================================================================#
    # RUN BATCHES
    #========================================================================#

    for batch_number, batch in enumerate(Repeat_Matrix):

        print("\n" + "#" * 70)
        print(
            f"BATCH {batch_number + 1} / "
            f"{len(Repeat_Matrix)}"
        )
        print("#" * 70)

        #====================================================================#
        # CREATE LANDSCAPES IN PARALLEL
        #====================================================================#

        print("\nCreating landscapes in parallel")

        landscape_args = [
            (
                repeat_number,
                repeat_dir,
                args.i
            )
            for repeat_number, repeat_dir in batch
        ]

        with ProcessPoolExecutor(
            max_workers=len(batch)
        ) as executor:

            futures = [
                executor.submit(
                    create_landscape,
                    landscape_arg
                )
                for landscape_arg in landscape_args
            ]

            for future in as_completed(futures):

                try:

                    repeat_number, repeat_dir = (
                        future.result()
                    )

                    print(
                        f"Finished landscape for "
                        f"Repeat_{repeat_number}"
                    )

                except Exception as e:

                    print(
                        f"ERROR creating landscape: {e}"
                    )

        print("Finished creating landscapes")

        #====================================================================#
        # RUN CDMetaPOP IN PARALLEL
        #====================================================================#

        print("\nRunning CDMetaPOP simulations")

        plist = []

        for repeat_number, repeat_dir in batch:

            print(
                f"Starting Repeat_{repeat_number}"
            )

            p = subprocess.Popen([
                "nice",
                "-n",
                "19",
                "uv",
                "run",
                "../../../src/CDmetaPOP.py",
                os.path.join(
                    repeat_dir,
                    "inputs"
                ),
                "RunVars.csv",
                "../outputs/raw/"
            ])

            plist.append(
                (repeat_number, p)
            )

        # Wait for all simulations in this batch
        for repeat_number, p in plist:

            returncode = p.wait()

            if returncode != 0:

                print(
                    f"WARNING: CDMetaPOP for "
                    f"Repeat_{repeat_number} "
                    f"finished with return code "
                    f"{returncode}"
                )

        print("Batch simulations finished")

        #====================================================================#
        # ANALYSE + DELETE IN PARALLEL
        #====================================================================#

        print(
            "\nAnalysing repeats in parallel"
        )

        with ProcessPoolExecutor(
            max_workers=len(batch)
        ) as executor:

            futures = {
                executor.submit(
                    analyse_repeat,
                    (repeat_number, repeat_dir)
                ): repeat_number
                for repeat_number, repeat_dir in batch
            }

            for future in as_completed(futures):

                repeat_number = futures[future]

                try:

                    result = future.result()

                    if result is None:

                        print(
                            f"Repeat_{repeat_number} "
                            f"returned no results"
                        )

                        continue

                    #========================================================#
                    # STORE RESULTS
                    #========================================================#

                    Exponential_Decay_parameters.append(
                        result["decay_parameter"]
                    )

                    heterozygosity_curves_list.append(
                        result["heterozygosity_curves"]
                    )

                    population_curves_list.append(
                        result["population_curves"]
                    )

                    matrixlist.append(
                        result["matrix"]
                    )

                    print(
                        f"Finished analysis for "
                        f"Repeat_{repeat_number}"
                    )

                except Exception as e:

                    print(
                        f"ERROR processing "
                        f"Repeat_{repeat_number}: {e}"
                    )

        print(
            f"\nFinished batch "
            f"{batch_number + 1} / "
            f"{len(Repeat_Matrix)}"
        )

    #========================================================================#
    # SAVE ANALYSIS RESULTS
    #========================================================================#

    results_path = os.path.join(
        SaveDirName,
        "Analysis_Results.npz"
    )

    np.savez(
        results_path,

        matrixlist=np.asarray(
            matrixlist,
            dtype=object
        ),

        Exponential_Decay_parameters=np.asarray(
            Exponential_Decay_parameters,
            dtype=object
        ),

        heterozygosity_curves_list=np.asarray(
            heterozygosity_curves_list,
            dtype=object
        ),

        population_curves_list=np.asarray(
            population_curves_list,
            dtype=object
        )
    )

    print(
        f"\nSaved analysis results to:"
    )

    print(results_path)

    #========================================================================#
    # FINISHED
    #========================================================================#

    print(
        "\nFinished all landscape simulations "
        "and analyses"
    )

    endtime = time.time()

    print(
        f"Total time taken: "
        f"{endtime - starttime:.2f} seconds"
    )


#=============================================================================#
# ENTRY POINT
#=============================================================================#

if __name__ == "__main__":
    main()
