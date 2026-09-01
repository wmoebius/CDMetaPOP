"""
This script calculates key statistics of Markov chain matrices.

The main() function takes a CDMAT matrix file and returns the
calculated statistics.

The script can also be run directly from the command line using:

    python Matrix_Analysis.py -d path/to/cdmatrix.csv
"""

import numpy as np
import argparse
from pathlib import Path


#=============================================================================#
# FUNCTIONS
#============================================================================#

def Mixing_Time(Dispersal_Matrix):

    eigenvalues = abs(
        np.linalg.eigvals(Dispersal_Matrix)
    )

    sorted_indices = np.argsort(eigenvalues)[::-1]

    second_largest = eigenvalues[sorted_indices[1]]

    mixing_time = 1 / (1 - second_largest)

    print("|Eigenvalues|:", eigenvalues)
    print("Second largest eigenvalue:", second_largest)
    print("Mixing time:", mixing_time)

    return mixing_time


def calculate_hub_statistics(ProbMatrix):
    """
    Calculate hub-related statistics for a row-stochastic
    transition/dispersal matrix.

    Parameters
    ----------
    ProbMatrix : np.ndarray
        Row-stochastic transition matrix P.
        P[i, j] = probability of moving from i to j.

    Returns
    -------
    stats : dict
        Dictionary containing hub statistics.
    """

    P = np.asarray(ProbMatrix, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(
            "ProbMatrix must be a square matrix."
        )

    #-------------------------------------------------------------------------#
    # Check that matrix is approximately row stochastic
    #-------------------------------------------------------------------------#

    row_sums = P.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError(
            "ProbMatrix must be row-stochastic: "
            "each row must sum to 1."
        )


    #-------------------------------------------------------------------------#
    # Incoming strength
    #-------------------------------------------------------------------------#

    incoming_strength = P.sum(axis=0)

    q = incoming_strength / incoming_strength.sum()


    #-------------------------------------------------------------------------#
    # Hub concentration and effective number of hubs
    #-------------------------------------------------------------------------#

    hub_concentration = np.sum(q**2)

    N_hub = 1.0 / hub_concentration


    #-------------------------------------------------------------------------#
    # Strongest hub
    #-------------------------------------------------------------------------#

    sorted_q = np.sort(q)[::-1]

    strongest_hub = sorted_q[0]

    cumulative_hub_contribution = np.cumsum(sorted_q)


    #-------------------------------------------------------------------------#
    # Stationary distribution
    #-------------------------------------------------------------------------#

    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    idx = np.argmin(
        np.abs(eigenvalues - 1.0)
    )

    pi = np.real(
        eigenvectors[:, idx]
    )

    if np.sum(pi) < 0:
        pi = -pi

    pi = pi / np.sum(pi)

    pi = np.maximum(pi, 0)

    pi = pi / np.sum(pi)


    #-------------------------------------------------------------------------#
    # Effective number of patches in stationary distribution
    #-------------------------------------------------------------------------#

    stationary_concentration = np.sum(pi**2)

    N_eff = 1.0 / stationary_concentration


    #-------------------------------------------------------------------------#
    # Return results
    #-------------------------------------------------------------------------#

    return {
        "incoming_strength": incoming_strength,
        "normalised_incoming_strength": q,

        "hub_concentration": hub_concentration,
        "effective_number_hubs": N_hub,

        "strongest_hub_fraction": strongest_hub,
        "sorted_hub_fractions": sorted_q,
        "cumulative_hub_contribution": cumulative_hub_contribution,

        "stationary_distribution": pi,
        "stationary_concentration": stationary_concentration,
        "effective_number_patches": N_eff,
    }


#=============================================================================#
# MAIN ANALYSIS FUNCTION
#============================================================================#

def main(cdmat_path):

    #-------------------------------------------------------------------------#
    # Check CDMAT file
    #-------------------------------------------------------------------------#

    cdmat_file = Path(cdmat_path)

    if not cdmat_file.is_file():

        raise FileNotFoundError(
            f"Not a file: {cdmat_file}"
        )


    #-------------------------------------------------------------------------#
    # Extract matrix
    #-------------------------------------------------------------------------#

    Dispersal_Matrix = np.genfromtxt(
        cdmat_file,
        delimiter=","
    )


    #-------------------------------------------------------------------------#
    # Calculate mixing time
    #-------------------------------------------------------------------------#

    mixing_time = Mixing_Time(
        Dispersal_Matrix
    )


    #-------------------------------------------------------------------------#
    # Calculate hub statistics
    #-------------------------------------------------------------------------#

    stats = calculate_hub_statistics(
        Dispersal_Matrix
    )

    print(
        "Effective number of hubs:",
        stats["effective_number_hubs"]
    )

    print(
        "Strongest hub fraction:",
        stats["strongest_hub_fraction"]
    )

    print(
        "Effective number of patches:",
        stats["effective_number_patches"]
    )


    #-------------------------------------------------------------------------#
    # Return all statistics
    #-------------------------------------------------------------------------#

    stats["mixing_time"] = mixing_time

    return stats


#=============================================================================#
# ARGPARSE
#
# This is ONLY used when Matrix_Analysis.py is run directly.
# It is NOT executed when main() is imported by another script.
#=============================================================================#

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate statistics for a CDMAT matrix"
    )

    parser.add_argument(
        "-d",
        type=str,
        required=True,
        help="Path to the CDMAT.csv file"
    )

    args = parser.parse_args()

    main(args.d)