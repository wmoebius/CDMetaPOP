import argparse
import os

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

args = parser.parse_args()


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
# LOAD
#=============================================================================#

with np.load(args.f, allow_pickle=True) as data:
    results = {key: data[key] for key in data.files}

heterozygosity_curves_list = results["heterozygosity_curves_list"]

print(
    f"Loaded {len(heterozygosity_curves_list)} heterozygosity curves "
    f"from {args.f}"
)


#=============================================================================#
# REFIT
#=============================================================================#

Exponential_Decay_parameters = []
Exponential_Decay_errors = []
Exponential_Decay_amplitudes = []
Exponential_Decay_amplitude_errors = []

for repeat_index, curve in enumerate(heterozygosity_curves_list):

    if len(curve) < args.n:
        print(
            f"WARNING: curve {repeat_index} has only {len(curve)} "
            f"generations, fewer than n = {args.n}; fitting over all of it"
        )

    a, a_error, H0, H0_error = fit_last_n(curve, args.n)

    Exponential_Decay_parameters.append(a)
    Exponential_Decay_errors.append(a_error)
    Exponential_Decay_amplitudes.append(H0)
    Exponential_Decay_amplitude_errors.append(H0_error)

    if args.plot:

        plot_refit(
            curve,
            a,
            a_error,
            H0,
            args.n,
            os.path.join(
                os.path.dirname(os.path.abspath(args.f)),
                f"refitted_heterozygosity_AVERAGE_landscape_average_"
                f"Landscape_{repeat_index}_last{args.n}.png"
            )
        )

Exponential_Decay_parameters = np.asarray(Exponential_Decay_parameters)

number_failed = np.count_nonzero(~np.isfinite(Exponential_Decay_parameters))

print(f"Refitted a over the last {args.n} generations")
print(f"  failed fits: {number_failed}")
print(f"  median a:    {np.nanmedian(Exponential_Decay_parameters):.3f}")


#=============================================================================#
# SAVE
#=============================================================================#

# Keep everything from the original file, replacing the decay parameters
# with the refitted ones so downstream analysis scripts work unchanged

results["Exponential_Decay_parameters_original"] = (
    results["Exponential_Decay_parameters"]
)

results["Exponential_Decay_parameters"] = np.asarray(
    Exponential_Decay_parameters,
    dtype=object
)

results["Exponential_Decay_errors"] = np.asarray(Exponential_Decay_errors)
results["Exponential_Decay_amplitudes"] = np.asarray(
    Exponential_Decay_amplitudes
)
results["Exponential_Decay_amplitude_errors"] = np.asarray(
    Exponential_Decay_amplitude_errors
)
results["fit_last_n"] = np.asarray(args.n)

root, extension = os.path.splitext(args.f)

output_path = f"{root}_last{args.n}{extension}"

np.savez(output_path, **results)

print(f"\nSaved refitted results to:")
print(output_path)
