import argparse
import os
import time

from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


#=============================================================================#
# ARGPARSER
#=============================================================================#

parser = argparse.ArgumentParser(
    prog="Refit_Decay_LastN.py",
    description=(
        "Refit the heterozygosity decay parameter a using only the last n "
        "generations of each landscape-wide heterozygosity curve in an "
        "Analysis_Results.npz"
    )
)

parser.add_argument(
    "-f",
    type=str,
    required=True,
    help="path to the Analysis_Results.npz to refit"
)

parser.add_argument(
    "-n",
    type=int,
    required=True,
    help="number of final generations to fit over"
)

parser.add_argument(
    "-plot",
    action="store_true",
    help="re-plot each landscape-wide heterozygosity curve with the refitted decay (default: False)"
)

parser.add_argument(
    "-ncores",
    type=int,
    default=1,
    help="number of cores to parallelise the refitting over (default: 1)"
)


#=============================================================================#
# FIT FUNCTION
#=============================================================================#

def fit_function(x, H0, a):
    # x is measured from the start of the fitting window, so H0 is the
    # heterozygosity at that point rather than the fixed 0.5 used when
    # fitting from generation 0
    return H0 * np.exp(-x / (2 * a))


def fit_last_n(curve, n):
    """
    Fit H0 * exp(-(t - t0) / (2a)) to the last n points of curve.

    Returns (a, a_error, H0, H0_error), with NaNs if the fit fails.
    """

    curve = np.asarray(curve, dtype=float)

    generations = np.arange(len(curve), dtype=float)

    generations = generations[-n:]
    curve = curve[-n:]

    valid = (
        np.isfinite(generations)
        & np.isfinite(curve)
        & (curve > 0)
    )

    if np.count_nonzero(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan

    x = generations[valid] - generations[0]
    y = curve[valid]

    # Initial guess from a straight-line fit to log(H)
    slope, intercept = np.polyfit(x, np.log(y), 1)

    H0_guess = np.exp(intercept)
    a_guess = -1 / (2 * slope) if slope < 0 else 1e6

    try:

        popt, pcov = curve_fit(
            fit_function,
            x,
            y,
            p0=[H0_guess, a_guess],
            maxfev=10000
        )

    except (RuntimeError, ValueError) as e:

        print(f"WARNING: fit failed: {e}")

        return np.nan, np.nan, np.nan, np.nan

    errors = np.sqrt(np.diag(pcov))

    return popt[1], errors[1], popt[0], errors[0]


def plot_refit(curve, a, a_error, H0, n, output_file):
    """
    Plot the full landscape-wide heterozygosity curve with the refitted
    decay over the last n generations, in the style of
    Heterozygosity_ai_alphabetagamma.fit_heterozygosity_curve.
    """

    curve = np.asarray(curve, dtype=float)

    generations = np.arange(len(curve), dtype=float)

    fit_generations = generations[-n:]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        generations,
        curve,
        linewidth=2,
        color="k",
        label="Average",
    )

    if np.isfinite(a):

        ax.plot(
            fit_generations,
            fit_function(
                fit_generations - fit_generations[0],
                H0,
                a,
            ),
            linestyle="dashed",
            label=(
                f"Fit: a = {a:.3f} +/- {a_error:.3f}"
            ),
        )

    ax.axvspan(
        fit_generations[0],
        fit_generations[-1],
        alpha=0.1,
        label=f"Fit window (last {n} generations)",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean heterozygosity")
    ax.set_title(
        f"Refitted landscape-wide average heterozygosity\n"
        f"Fit over last {n} generations"
    )

    ax.set_xlim(
        generations[0],
        generations[-1],
    )

    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


#=============================================================================#
# WORKER FUNCTION
#=============================================================================#

def refit_curve(args):
    """
    Refit (and optionally plot) one curve.

    This function is run in a separate process. Returns
    (repeat_index, a, a_error, H0, H0_error).
    """

    repeat_index, curve, n, plot, plot_dir = args

    if len(curve) < n:
        print(
            f"  WARNING: curve {repeat_index} has only {len(curve)} "
            f"generations, fewer than n = {n}; fitting over all of it",
            flush=True
        )

    a, a_error, H0, H0_error = fit_last_n(curve, n)

    print(
        f"  Curve index {repeat_index}: a = {a:.3f} +/- {a_error:.3f}",
        flush=True
    )

    if plot:

        plot_refit(
            curve,
            a,
            a_error,
            H0,
            n,
            os.path.join(
                plot_dir,
                f"refitted_heterozygosity_AVERAGE_landscape_average_"
                f"Landscape_{repeat_index}_last{n}.png"
            )
        )

    return repeat_index, a, a_error, H0, H0_error


#=============================================================================#
# MAIN
#=============================================================================#

def main():

    args = parser.parse_args()

    #========================================================================#
    # LOAD
    #========================================================================#

    starttime = time.time()

    print(
        f"Loading {args.f} "
        f"({os.path.getsize(args.f) / 1e9:.2f} GB)",
        flush=True
    )

    results = {}

    with np.load(args.f, allow_pickle=True) as data:

        print(f"  Arrays in file: {data.files}", flush=True)

        for key in data.files:

            print(f"  Loading {key} ...", flush=True)

            key_starttime = time.time()

            results[key] = data[key]

            print(
                f"  Loaded {key}: shape {results[key].shape}, "
                f"{time.time() - key_starttime:.1f} s",
                flush=True
            )

    heterozygosity_curves_list = results["heterozygosity_curves_list"]

    number_curves = len(heterozygosity_curves_list)

    print(
        f"Finished loading: {number_curves} heterozygosity "
        f"curves ({time.time() - starttime:.1f} s)",
        flush=True
    )

    #========================================================================#
    # SPLIT INTO BATCHES
    #========================================================================#

    indexed_curves = list(enumerate(heterozygosity_curves_list))

    Curve_Batches = [
        indexed_curves[i:i + args.ncores]
        for i in range(0, number_curves, args.ncores)
    ]

    print(
        f"\nRefitting over the last {args.n} generations: "
        f"{number_curves} curves in {len(Curve_Batches)} batches "
        f"of maximum size {args.ncores}",
        flush=True
    )

    #========================================================================#
    # REFIT BATCH BY BATCH
    #========================================================================#

    Exponential_Decay_parameters = np.full(number_curves, np.nan)
    Exponential_Decay_errors = np.full(number_curves, np.nan)
    Exponential_Decay_amplitudes = np.full(number_curves, np.nan)
    Exponential_Decay_amplitude_errors = np.full(number_curves, np.nan)

    plot_dir = os.path.dirname(os.path.abspath(args.f))

    # One pool for the whole run; each batch of ncores curves is submitted
    # together and finished before the next batch starts

    with ProcessPoolExecutor(
        max_workers=args.ncores
    ) as executor:

        for batch_number, batch in enumerate(Curve_Batches):

            print(
                f"\nBATCH {batch_number + 1} / {len(Curve_Batches)}",
                flush=True
            )

            futures = {
                executor.submit(
                    refit_curve,
                    (repeat_index, curve, args.n, args.plot, plot_dir)
                ): repeat_index
                for repeat_index, curve in batch
            }

            for future in as_completed(futures):

                repeat_index = futures[future]

                try:

                    repeat_index, a, a_error, H0, H0_error = (
                        future.result()
                    )

                except Exception as e:

                    print(
                        f"ERROR refitting curve index {repeat_index}: {e}",
                        flush=True
                    )

                    continue

                # Store by curve index so the output keeps the input order
                Exponential_Decay_parameters[repeat_index] = a
                Exponential_Decay_errors[repeat_index] = a_error
                Exponential_Decay_amplitudes[repeat_index] = H0
                Exponential_Decay_amplitude_errors[repeat_index] = H0_error

    number_failed = np.count_nonzero(
        ~np.isfinite(Exponential_Decay_parameters)
    )

    print(f"\nRefitted a over the last {args.n} generations")
    print(f"  failed fits: {number_failed}")
    print(f"  median a:    {np.nanmedian(Exponential_Decay_parameters):.3f}")

    #========================================================================#
    # SAVE
    #========================================================================#

    # Keep everything from the original file, replacing the decay parameters
    # with the refitted ones so downstream analysis scripts work unchanged

    results["Exponential_Decay_parameters_original"] = (
        results["Exponential_Decay_parameters"]
    )

    results["Exponential_Decay_parameters"] = np.asarray(
        Exponential_Decay_parameters,
        dtype=object
    )

    results["Exponential_Decay_errors"] = Exponential_Decay_errors
    results["Exponential_Decay_amplitudes"] = Exponential_Decay_amplitudes
    results["Exponential_Decay_amplitude_errors"] = (
        Exponential_Decay_amplitude_errors
    )
    results["fit_last_n"] = np.asarray(args.n)

    root, extension = os.path.splitext(args.f)

    output_path = f"{root}_last{args.n}{extension}"

    print(f"\nSaving refitted results to:\n{output_path}", flush=True)

    np.savez(output_path, **results)

    print("Finished saving", flush=True)
    print(f"Total time taken: {time.time() - starttime:.1f} s")


#=============================================================================#
# ENTRY POINT
#=============================================================================#

if __name__ == "__main__":
    main()
