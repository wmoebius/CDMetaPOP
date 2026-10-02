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
import Heterozygosity_ai_alphabetagamma

import time
import subprocess
import os
import shutil
import argparse
import copy

from concurrent.futures import ProcessPoolExecutor, as_completed


#=============================================================================#
# CONSTANTS
#=============================================================================#

MAX_LANDSCAPE_ATTEMPTS = 3
LANDSCAPE_RETRY_DELAY_SECONDS = 5


#=============================================================================#
# WORKER FUNCTIONS
#=============================================================================#

def create_landscape(args):
    """
    Create one landscape, retrying on failure.

    This function is run in a separate process. Returns
    (rep, directoryname, success, error_message).
    """

    rep, directoryname, input_dir,Global_RandomSeed = args

    last_error = None

    for attempt in range(1, MAX_LANDSCAPE_ATTEMPTS + 1):

        if attempt > 1:

            print(
                f"Retrying landscape for Repeat_{rep} "
                f"(attempt {attempt}/{MAX_LANDSCAPE_ATTEMPTS})"
            )

            shutil.rmtree(directoryname, ignore_errors=True)

            time.sleep(LANDSCAPE_RETRY_DELAY_SECONDS)

        else:

            print(f"Creating landscape for Repeat_{rep}")

        try:

            Landscape_Construction.main(
                n=n,
                ProbDist=ProbDist,
                param1=param1,
                d=directoryname,
                i=input_dir,
                r=rep + Global_RandomSeed
            )

            return rep, directoryname, True, None

        except Exception as e:

            last_error = str(e)

            print(
                f"ERROR creating landscape for Repeat_{rep} "
                f"(attempt {attempt}/{MAX_LANDSCAPE_ATTEMPTS}): {e}"
            )

    return rep, directoryname, False, last_error


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

    matrix = np.transpose(matrix)

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

        decay_parameter, heterozygosity_curves, population_curves, alpha_heterozygosity_curves, beta_heterozygosity_matrices = (
            Heterozygosity_ai_alphabetagamma.main(
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
        "alpha_heterozygosity_curves": copy.copy(
            alpha_heterozygosity_curves
        ),
        "beta_heterozygosity_matrices": copy.copy(
            beta_heterozygosity_matrices
        )
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


    parser.add_argument(
        "-random",
        action="store_true",
        help='Instantiate random number generator with a random seed (default: False)'
    )

    args = parser.parse_args()

    #========================================================================#
    # SAVING DETAILS
    #========================================================================#

    Global_RandomSeed = 0

    if args.random:

        Global_RandomSeed = np.random.randint(0, 2**32 - 1)

        SaveDirName += "_randomseed_%d" %(Global_RandomSeed)



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

    alpha_heterozygosity_curves_list = []
    beta_heterozygosity_matrices_list = []

    all_failed_repeats = []

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
                args.i,
                Global_RandomSeed
            )
            for repeat_number, repeat_dir in batch
        ]

        successful_batch = []
        failed_repeat_numbers = []

        with ProcessPoolExecutor(
            max_workers=len(batch)
        ) as executor:

            futures = {
                executor.submit(
                    create_landscape,
                    landscape_arg
                ): landscape_arg[0]
                for landscape_arg in landscape_args
            }

            for future in as_completed(futures):

                repeat_number = futures[future]

                try:

                    repeat_number, repeat_dir, success, error_message = (
                        future.result()
                    )

                    if success:

                        successful_batch.append(
                            (repeat_number, repeat_dir)
                        )

                        print(
                            f"Finished landscape for "
                            f"Repeat_{repeat_number}"
                        )

                    else:

                        failed_repeat_numbers.append(repeat_number)

                        print(
                            f"Repeat_{repeat_number} landscape creation "
                            f"failed after {MAX_LANDSCAPE_ATTEMPTS} "
                            f"attempts: {error_message}"
                        )

                        all_failed_repeats.append(
                            (repeat_number, error_message)
                        )

                except Exception as e:

                    print(
                        f"ERROR creating landscape for "
                        f"Repeat_{repeat_number}: {e}"
                    )

                    failed_repeat_numbers.append(repeat_number)
                    all_failed_repeats.append(
                        (repeat_number, str(e))
                    )

        if failed_repeat_numbers:

            print(
                f"WARNING: {len(failed_repeat_numbers)} repeat(s) skipped "
                f"after {MAX_LANDSCAPE_ATTEMPTS} attempts: "
                f"{sorted(failed_repeat_numbers)}"
            )

        print("Finished creating landscapes")

        if not successful_batch:

            print(
                "Entire batch failed landscape creation "
                "- skipping simulation and analysis for this batch"
            )

            continue

        #====================================================================#
        # RUN CDMetaPOP IN PARALLEL
        #====================================================================#

        print("\nRunning CDMetaPOP simulations")

        plist = []

        for repeat_number, repeat_dir in successful_batch:

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
            max_workers=len(successful_batch)
        ) as executor:

            futures = {
                executor.submit(
                    analyse_repeat,
                    (repeat_number, repeat_dir)
                ): repeat_number
                for repeat_number, repeat_dir in successful_batch
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

                    alpha_heterozygosity_curves_list.append(
                        result["alpha_heterozygosity_curves"]
                    )

                    beta_heterozygosity_matrices_list.append(
                        result["beta_heterozygosity_matrices"]
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
        ),

        alpha_heterozygosity_curves_list=np.asarray(
            alpha_heterozygosity_curves_list,
            dtype=object
        ),
        beta_heterozygosity_matrices_list=np.asarray(
            beta_heterozygosity_matrices_list,
            dtype=object
        )
    )

    print(
        f"\nSaved analysis results to:"
    )

    print(results_path)

    #========================================================================#
    # FAILED REPEATS LOG
    #========================================================================#

    if all_failed_repeats:

        failed_repeats_path = os.path.join(
            SaveDirName,
            "failed_repeats.txt"
        )

        with open(failed_repeats_path, "w") as failed_repeats_file:

            for repeat_number, error_message in sorted(
                all_failed_repeats,
                key=lambda item: item[0]
            ):

                failed_repeats_file.write(
                    f"Repeat_{repeat_number}: {error_message}\n"
                )

        print(
            f"\n{len(all_failed_repeats)} repeat(s) permanently failed "
            f"landscape creation after {MAX_LANDSCAPE_ATTEMPTS} attempts "
            f"each. Details written to:"
        )

        print(failed_repeats_path)

    else:

        print(
            "\nAll repeats succeeded landscape creation"
        )

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
