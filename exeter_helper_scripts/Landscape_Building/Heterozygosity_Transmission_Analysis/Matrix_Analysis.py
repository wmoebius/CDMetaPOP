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
from itertools import combinations
import networkx as nx

#=============================================================================#
# FUNCTIONS
#============================================================================#

def Mixing_Time(Dispersal_Matrix):

    """
    Calculate the mixing time of a Markov transition matrix, 
    which is the number of steps required for the Markov chain to converge 
    to its stationary distribution from any initial distribution.
    """

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



def Kemeny_Constant(Dispersal_Matrix):
    """
    Calculate the Kemeny constant of a Markov transition matrix.

    The Kemeny constant is a measure of the expected time to reach
    a randomly chosen state in the stationary distribution from a given starting state, averaged over
    all starting states. The larger the value, the longer it takes to reach a randomly chosen state on average.

    Parameters
    ----------
    Dispersal_Matrix : numpy.ndarray
        2D row-stochastic transition matrix.

    Returns
    -------
    float
        Kemeny constant.
    """

    P = np.asarray(Dispersal_Matrix, dtype=float)

    # Basic checks
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Dispersal_Matrix must be a square 2D matrix.")

    if not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Dispersal_Matrix must be row-stochastic.")

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(P)

    # Remove the eigenvalue corresponding to lambda = 1
    eigenvalues = eigenvalues[np.abs(eigenvalues - 1) > 1e-10]

    # Kemeny's constant
    K = np.sum(1 / (1 - eigenvalues))

    return float(np.real(K))


def Conductance(Dispersal_Matrix):
    """
    Calculate the conductance of a Markov transition matrix. 
    A high conductance indicates that the Markov chain mixes quickly, 
    while a low conductance suggests that there are bottlenecks or "hubs" 
    in the state space that slow down mixing.

    Parameters
    ----------
    Dispersal_Matrix : numpy.ndarray
        2D row-stochastic transition matrix.

    Returns
    -------
    float
        Conductance of the Markov chain.
    """

    P = np.asarray(Dispersal_Matrix, dtype=float)

    # Basic checks
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Dispersal_Matrix must be a square 2D matrix.")

    if not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Dispersal_Matrix must be row-stochastic.")

    n = P.shape[0]

    # Find stationary distribution:
    # pi P = pi
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    idx = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, idx])

    # Ensure positive orientation
    pi = np.abs(pi)
    pi /= pi.sum()

    best_conductance = np.inf

    # Enumerate subsets up to half the stationary probability
    for r in range(1, n):
        for subset in combinations(range(n), r):

            S = np.array(subset)

            # Stationary probability of S
            pi_S = pi[S].sum()

            if pi_S > 0.5 + 1e-12:
                continue

            # Probability flow from S to S^c
            mask = np.ones(n, dtype=bool)
            mask[S] = False
            Sc = np.where(mask)[0]

            flow = np.sum(
                pi[S, None] * P[np.ix_(S, Sc)]
            )

            phi = flow / pi_S

            best_conductance = min(best_conductance, phi)

    return float(best_conductance)




def mean_effective_degree(Dispersal_Matrix):
    """
    Mean effective number of dispersal destinations.

    Higher values mean dispersal is spread more evenly across
    a larger number of patches.
    """

    P = np.asarray(Dispersal_Matrix, dtype=float)

    Entropy = np.zeros(P.shape[0])

    for i in range(P.shape[0]):

        Row = P[i]

        Nonzero = Row > 0

        Entropy[i] = -np.sum(
            Row[Nonzero] *
            np.log(Row[Nonzero])
        )

    Effective_Degree = np.exp(Entropy)

    return float(np.mean(Effective_Degree))


def cv_effective_degree(Dispersal_Matrix):
    """
    Coefficient of variation of effective dispersal degree.

    Higher values mean greater heterogeneity between patches
    in how broadly they disperse.
    """

    P = np.asarray(Dispersal_Matrix, dtype=float)

    Entropy = np.zeros(P.shape[0])

    for i in range(P.shape[0]):

        Row = P[i]

        Nonzero = Row > 0

        Entropy[i] = -np.sum(
            Row[Nonzero] *
            np.log(Row[Nonzero])
        )

    Effective_Degree = np.exp(Entropy)

    Mean = np.mean(Effective_Degree)

    if Mean == 0:
        return np.nan

    return float(
        np.std(Effective_Degree) / Mean
    )


def cv_stationary_distribution(Dispersal_Matrix):
    """
    Coefficient of variation of the stationary distribution.

    Higher values mean that the long-term movement process is
    concentrated more strongly in some patches than others.
    """

    P = np.asarray(Dispersal_Matrix, dtype=float)

    Eigenvalues, Eigenvectors = np.linalg.eig(P.T)

    Index = np.argmin(
        np.abs(Eigenvalues - 1)
    )

    Stationary_Distribution = np.real(
        Eigenvectors[:, Index]
    )

    # Remove arbitrary eigenvector sign
    Stationary_Distribution = np.abs(
        Stationary_Distribution
    )

    Stationary_Distribution /= np.sum(
        Stationary_Distribution
    )

    Mean = np.mean(
        Stationary_Distribution
    )

    if Mean == 0:
        return np.nan

    return float(
        np.std(Stationary_Distribution) / Mean
    )

def mean_clustering(Dispersal_Matrix):
    """
    Mean weighted clustering coefficient.

    Higher values indicate that patches connected to a given
    patch also tend to be connected to one another.

    The dispersal matrix is symmetrised because clustering is
    being used here as a measure of spatial neighbourhood
    structure rather than directed dispersal.
    """

    P = np.asarray(Dispersal_Matrix, dtype=float)

    Symmetric_P = (
        P + P.T
    ) / 2

    G = nx.from_numpy_array(
        Symmetric_P
    )

    Clustering = np.array(
        list(
            nx.clustering(
                G,
                weight="weight"
            ).values()
        )
    )

    return float(
        np.mean(Clustering)
    )
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

    """
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
    # Calculate Kemeny's constant
    #-------------------------------------------------------------------------#

    kemeny_constant = Kemeny_Constant(Dispersal_Matrix)

    print("Kemeny's constant:",kemeny_constant)


    #-------------------------------------------------------------------------#
    # Calculate conductance
    #-------------------------------------------------------------------------#
    conductance = Conductance(Dispersal_Matrix)
    print("Conductance:", conductance)


    #-------------------------------------------------------------------------#
    # Return all statistics
    #-------------------------------------------------------------------------#
    """
    stats = {

    "mixing_time": Mixing_Time(Dispersal_Matrix),

    "kemeny_constant":
        Kemeny_Constant(Dispersal_Matrix),

    "conductance":
        Conductance(Dispersal_Matrix),

    "mean_effective_degree":
        mean_effective_degree(Dispersal_Matrix),

    "cv_effective_degree":
        cv_effective_degree(Dispersal_Matrix),

    "cv_stationary_distribution":
        cv_stationary_distribution(Dispersal_Matrix),

    "mean_clustering":
        mean_clustering(Dispersal_Matrix)
}

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