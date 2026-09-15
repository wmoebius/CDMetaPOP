import numpy as np
import os
import argparse
import copy
import time

from Matrix_Analysis import Mixing_Time
import matplotlib.pyplot as plt


starttime = time.time()

#=============================================================================#
# ARGPARSER
#=============================================================================#

parser = argparse.ArgumentParser(
    description="Check saved analysis results"
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing Analysis_Results.npz"
)

parser.add_argument(
    "-shufflenum",
    type=int,
    default=1,
    help="Number of times to shuffle the transition matrices"
)

parser.add_argument(
    "--analysis-mode",
    action = 'store_true',
    help = "Whether or not to open in analysis mode"
)

args = parser.parse_args()


#=============================================================================#
# LOAD RESULTS
#=============================================================================#

results_path = os.path.join(
    args.d,
    "Analysis_Results.npz"
)

if not os.path.isfile(results_path):
    raise FileNotFoundError(
        f"Could not find Analysis_Results.npz in: {args.d}"
    )

data = np.load(
    results_path,
    allow_pickle=True
)

matrixlist = data["matrixlist"].tolist()

Exponential_Decay_parameters = (
    data["Exponential_Decay_parameters"].tolist()
)

#heterozygosity_curves_list = data["heterozygosity_curves_list"].tolist()

#population_curves_list = data["population_curves_list"].tolist()

#=============================================================================#
# PRINT RESULTS
#=============================================================================#

print("\n" + "=" * 70)
print("matrixlist")
print("=" * 70)

print(matrixlist[0])


print("\n" + "=" * 70)
print("EXPONENTIAL DECAY PARAMETERS")
print("=" * 70)

print(Exponential_Decay_parameters)

"""
print("\n" + "=" * 70)
print("HETEROZYGOSITY CURVES LIST")
print("=" * 70)

print(heterozygosity_curves_list)


print("\n" + "=" * 70)
print("Number CURVES LIST")
print("=" * 70)

print(population_curves_list)
"""

import numpy as np

#DELETE BELOWuuuu

def heterozygosity_a(T, K=100):
    """
    Calculate a for heterozygosity decay on a network, where
    H(t) ~ H_0 * exp(-t / (2a)).

    T[i,j] = forward probability of moving from node i to node j.

    Assumes:
        - diploid Wright-Fisher reproduction
        - K is the mean per-node carrying capacity (total
          metapopulation size = n*K); actual per-node equilibrium
          size is not flat K, but whatever migration in/out implies
          (e.g. via CDMetaPOP's breeding/death dynamics), estimated
          here from the stationary distribution of T
        - neutral evolution
        - migration described by T
        - arbitrary (non-symmetric) T

    Returns
    -------
    a : float
        The predicted heterozygosity decay parameter a.

    """

    T = np.asarray(T, dtype=float)

    n = T.shape[0]

    if T.shape != (n, n):
        raise ValueError("T must be square.")

    # Normalise rows
    T = T / T.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------
    # Stationary distribution
    #
    # Under stable (if uneven) node populations, flow balance
    # requires N_i = sum_k N_k * T[k,i], i.e. relative node sizes
    # are the left eigenvector of T with eigenvalue 1.
    # ---------------------------------------------------------

    eigvals, eigvecs = np.linalg.eig(T.T)

    idx = np.argmin(np.abs(eigvals - 1.0))

    pi = np.real(eigvecs[:, idx])

    # Eigenvectors can have arbitrary sign
    pi = np.abs(pi)
    pi /= pi.sum()

    # ---------------------------------------------------------
    # Backwards (ancestral) transition matrix
    #
    # B[i,j] = probability that the parent of an individual
    #          in i was in j.
    # ---------------------------------------------------------

    B = np.zeros_like(T)

    for i in range(n):
        for j in range(n):
            B[i, j] = pi[j] * T[j, i] / pi[i]

    # ---------------------------------------------------------
    # Two-lineage transition matrix
    # ---------------------------------------------------------

    Q = np.zeros((n*n, n*n))

    N_total = n * K
    N = pi * N_total

    def state(i, j):
        return i*n + j

    for i in range(n):
        for j in range(n):

            row = state(i, j)

            for k in range(n):
                for l in range(n):

                    p = B[i, k] * B[j, l]

                    # If both lineages are in the same deme,
                    # they coalesce with probability 1/(2*N_k),
                    # where N_k is that deme's implied equilibrium
                    # size (not flat K).
                    if k == l:
                        p *= (1.0 - 1.0/(2.0*N[k]))

                    Q[row, state(k, l)] += p

    # ---------------------------------------------------------
    # Dominant eigenvalue
    # ---------------------------------------------------------

    eigenvalues = np.linalg.eigvals(Q)

    lambda_max = np.max(np.real(eigenvalues))

    # H(t) ~ H_0 * lambda_max**t = H_0 * exp(-t/(2a))
    a = -1.0 / (2.0 * np.log(lambda_max))

    return a

if args.analysis_mode:
    mixing_times = []

    for i in range(len(matrixlist)):
        print(i)
        mixing_time = Mixing_Time(matrixlist[i])
        mixing_times.append(mixing_time)

    plt.scatter(mixing_times, Exponential_Decay_parameters)
    plt.xscale('log')
    plt.yscale("log")
    plt.xlabel("Mixing time")
    plt.ylabel("Exponential decay parameters")
    plt.title("Mixing time vs Exponential decay parameters")
    plt.savefig(str(args.d) + "/MixingTime_Versus_Decay.png")
    plt.show()


###############################################################################
#Generate new data files
###############################################################################

#First: rearrange each transition matrix such that the nodes are ordered by in-flow (the column sums)

def sort_TM_ByColSum(T):
    """
    Sort a transition matrix according to total input weight.

    Parameters
    ----------
    T : np.ndarray
        Square transition matrix, shape (N, N).
        T[i, j] is the transition probability from node i to node j.

    Returns
    -------
    T_sorted : np.ndarray
        Transition matrix with nodes reordered from highest
        total input weight to lowest.

    order : np.ndarray
        Indices giving the new node ordering.

    input_weights : np.ndarray
        Original total input weight of each node.
    """

    # Total weight entering each node
    input_weights = T.sum(axis=0)

    # Nodes ordered from highest input weight to lowest
    order = np.argsort(input_weights)[::-1]

    # Reorder both rows and columns
    T_sorted = T[np.ix_(order, order)]

    output_weights = T_sorted.sum(axis=0)

    return T_sorted, order, input_weights, output_weights



Sorted_matrixlist = []
for TM in matrixlist:
    T_sorted, order, input_weights, output_weights = sort_TM_ByColSum(np.asarray(TM))
    Sorted_matrixlist.append(T_sorted)
    #print(T_sorted)
    #print(order)
    #print(input_weights)
    #print(output_weights)

results_path = os.path.join(
    str(args.d),
    "Analysis_Results_Sorted.npz"
)

np.savez(
    results_path,
    matrixlist=np.asarray(Sorted_matrixlist, dtype=object),
    Exponential_Decay_parameters=np.asarray(
        Exponential_Decay_parameters,
        dtype=object
    )
)




#Second: randomly shuffle the order of the nodes in each transition matrix

def shuffle_transition_matrix(T):
    """
    Randomly shuffle the node labels of a transition matrix.

    Parameters
    ----------
    T : np.ndarray
        Square transition matrix, shape (N, N).
        T[i, j] is the transition probability from node i to node j.

    Returns
    -------
    T_shuffled : np.ndarray
        Transition matrix with rows and columns randomly reordered.

    order : np.ndarray
        The permutation used to reorder the nodes.
    """

    # Generate a random permutation of the node indices
    order = np.random.permutation(T.shape[0])

    # Apply the same permutation to rows and columns
    T_shuffled = T[np.ix_(order, order)]

    return T_shuffled, order


Shuffled_MatrixList = []
Exponential_Decay_parameters_shuffle = []

for i in range(len(matrixlist)):

    TM = matrixlist[i]
    Exponential_Decay_parameter = Exponential_Decay_parameters[i]

    for j in range(args.shufflenum):
        T_shuffled, order = shuffle_transition_matrix(np.asarray(TM))
        Shuffled_MatrixList.append(T_shuffled)
        Exponential_Decay_parameters_shuffle.append(Exponential_Decay_parameter)
    #print(T_shuffled.sum(axis=1))  # Check that column sums are still 1
    #print(order)

results_path = os.path.join(
    str(args.d),
    "Analysis_Results_Shuffled.npz"
)

np.savez(
    results_path,
    matrixlist=np.asarray(Shuffled_MatrixList, dtype=object),
    Exponential_Decay_parameters=np.asarray(
        Exponential_Decay_parameters_shuffle,
        dtype=object
    )
)

endtime = time.time()

print("Time taken:",endtime-starttime)
