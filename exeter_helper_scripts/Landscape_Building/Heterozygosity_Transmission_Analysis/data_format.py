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

heterozygosity_curves_list = data["heterozygosity_curves_list"].tolist()

population_curves_list = data["population_curves_list"].tolist()

alpha_heterozygosity_curves_list = data["alpha_heterozygosity_curves_list"].tolist()


beta_heterozygosity_matrices_list = data["beta_heterozygosity_matrices_list"].tolist()

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

if args.analysis_mode:

    """
    Calculate mixing times for each transition matrix and plot against the
    corresponding exponential decay parameters.
    """
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
    plt.close()


    """
    For each landscape, plot all alpha heterozygosity curves
    for each patch, with one curve per repeat.
    """
    """
    for i in range(len(alpha_heterozygosity_curves_list)):

        n_repeats = len(alpha_heterozygosity_curves_list[i])
        
        # First list is empty, so subtract one
        n_patches = len(alpha_heterozygosity_curves_list[i][0]) - 1

        for j in range(n_patches):

            plt.figure(figsize=(8, 6))

            # j = 0 corresponds to Patch 1
            patch_index = j + 1

            # Plot this patch for every repeat
            for k in range(n_repeats):

                curve = alpha_heterozygosity_curves_list[i][k][patch_index]

                plt.plot(
                    np.arange(len(curve)),
                    curve,
                    label=f"Repeat {k+1}"
                )

            plt.title(
                f"Alpha Heterozygosity Curves "
                f"for Landscape {i}, Patch {j+1}"
            )
            plt.xlabel("Time")
            plt.ylabel("Alpha Heterozygosity")
            plt.legend()

            plt.savefig(
                str(args.d)
                + f"/Alpha_Heterozygosity_Curves_Landscape_{i}, Patch_{j+1}.png"
            )

            plt.close()
    """

    """
    For each landscape, plot the beta heterozygosity matrix
    for each repeat.
    """
    #For each landscape
    for i in range(len(beta_heterozygosity_matrices_list)):

        n_repeats = len(beta_heterozygosity_matrices_list[i])

        #for each repeat
        for j in range(n_repeats):

            # Beta matrix for this landscape and repeat, LAST ONE only
            matrix = np.asarray(beta_heterozygosity_matrices_list[i][j][-1])
            print(np.shape(matrix))
            plt.figure(figsize=(8, 6))

            plt.imshow(
                matrix,
                cmap="viridis",
                interpolation="nearest"
            )

            plt.colorbar(label="Beta Heterozygosity")

            plt.title(
                f"Beta Heterozygosity Matrix "
                f"for Landscape {i}, Repeat {j+1}"
            )

            # Matrix indices 0,...,19 correspond to Patch IDs 1,...,20
            plt.xlabel("Patch Index")
            plt.ylabel("Patch Index")

            plt.xticks(
                np.arange(matrix.shape[1]),
                np.arange(1, matrix.shape[1] + 1)
            )
            plt.yticks(
                np.arange(matrix.shape[0]),
                np.arange(1, matrix.shape[0] + 1)
            )

            plt.savefig(
                str(args.d)
                + f"/Beta_Heterozygosity_Matrix_Landscape_{i}, "
                f"Repeat_{j+1}.png"
            )

            plt.close()

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


#=============================================================================#
# REORDER HETEROZYGOSITY DATA
#=============================================================================#

def reorder_alpha_heterozygosity(alpha_data, order, has_dummy_index=True):
    """
    Reorder alpha heterozygosity according to a node permutation.

    Parameters
    ----------
    alpha_data : list
        Alpha data for one landscape:
        [repeat][patch][time]

        If has_dummy_index=True, patch index 0 is assumed to be
        an empty/dummy entry, with patch 1 at index 1.

    order : np.ndarray
        Permutation of the transition-matrix node indices.

    has_dummy_index : bool
        Whether alpha_data has an unused element at index 0.

    Returns
    -------
    alpha_reordered : list
        Alpha heterozygosity with patch labels reordered.
    """

    alpha_reordered = []

    for repeat in alpha_data:

        if has_dummy_index:
            # Matrix node 0 corresponds to patch 1,
            # matrix node 1 corresponds to patch 2, etc.
            patches = repeat[1:]

            reordered_patches = [
                patches[i]
                for i in order
            ]

            # Put dummy element back at index 0
            reordered_repeat = [repeat[0]] + reordered_patches

        else:
            reordered_repeat = [
                repeat[i]
                for i in order
            ]

        alpha_reordered.append(reordered_repeat)

    return alpha_reordered


def reorder_beta_heterozygosity(beta_data, order):
    """
    Reorder beta heterozygosity according to a node permutation.

    Parameters
    ----------
    beta_data : list
        Beta data for one landscape and one repeat:
        [time][patch][patch]

    order : np.ndarray
        Permutation of the transition-matrix node indices.

    Returns
    -------
    beta_reordered : list
        Beta heterozygosity with both patch axes reordered.
    """

    beta_reordered = []

    for matrix in beta_data:

        matrix = np.asarray(matrix)

        # Reorder both patch dimensions
        matrix_reordered = matrix[np.ix_(order, order)]

        beta_reordered.append(matrix_reordered)

    return beta_reordered



#=============================================================================#
# SORT DATA BY TRANSITION-MATRIX COLUMN SUM
#=============================================================================#

Sorted_matrixlist = []
Sorted_alpha_heterozygosity = []
Sorted_beta_heterozygosity = []

for i in range(len(matrixlist)):

    # ---------------------------------------------------------
    # Sort transition matrix
    # ---------------------------------------------------------

    T_sorted, order, input_weights, output_weights = (
        sort_TM_ByColSum(np.asarray(matrixlist[i]))
    )

    Sorted_matrixlist.append(T_sorted)

    # ---------------------------------------------------------
    # Apply exactly the same node permutation to alpha data
    # for every repeat of this landscape
    # ---------------------------------------------------------

    alpha_sorted = reorder_alpha_heterozygosity(
        alpha_heterozygosity_curves_list[i],
        order,
        has_dummy_index=True
    )

    Sorted_alpha_heterozygosity.append(alpha_sorted)

    # ---------------------------------------------------------
    # Apply exactly the same node permutation to beta data
    # for every repeat and every time point
    # ---------------------------------------------------------

    beta_sorted = []

    for repeat in beta_heterozygosity_matrices_list[i]:

        beta_sorted.append(
            reorder_beta_heterozygosity(
                repeat,
                order
            )
        )

    Sorted_beta_heterozygosity.append(beta_sorted)


#=============================================================================#
# SAVE SORTED DATA
#=============================================================================#

results_path = os.path.join(
    str(args.d),
    "Analysis_Results_Sorted.npz"
)

np.savez(
    results_path,

    matrixlist=np.asarray(
        Sorted_matrixlist,
        dtype=object
    ),

    Exponential_Decay_parameters=np.asarray(
        Exponential_Decay_parameters,
        dtype=object
    ),

    alpha_heterozygosity_curves_list=np.asarray(
        Sorted_alpha_heterozygosity,
        dtype=object
    ),

    beta_heterozygosity_matrices_list=np.asarray(
        Sorted_beta_heterozygosity,
        dtype=object
    )
)
"""
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
"""



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


#=============================================================================#
# RANDOMLY SHUFFLE NODE LABELS
#=============================================================================#

Shuffled_MatrixList = []
Shuffled_Alpha_Heterozygosity = []
Shuffled_Beta_Heterozygosity = []
Exponential_Decay_parameters_shuffle = []

for i in range(len(matrixlist)):

    TM = matrixlist[i]

    Exponential_Decay_parameter = (
        Exponential_Decay_parameters[i]
    )

    for j in range(args.shufflenum):

        # ---------------------------------------------------------
        # Generate one random permutation
        # ---------------------------------------------------------

        T_shuffled, order = shuffle_transition_matrix(
            np.asarray(TM)
        )

        # ---------------------------------------------------------
        # Store shuffled transition matrix
        # ---------------------------------------------------------

        Shuffled_MatrixList.append(T_shuffled)

        # ---------------------------------------------------------
        # Store corresponding decay parameter
        # ---------------------------------------------------------

        Exponential_Decay_parameters_shuffle.append(
            Exponential_Decay_parameter
        )

        # ---------------------------------------------------------
        # Shuffle alpha heterozygosity
        #
        # Same permutation for every repeat of this landscape
        # ---------------------------------------------------------

        alpha_shuffled = reorder_alpha_heterozygosity(
            alpha_heterozygosity_curves_list[i],
            order,
            has_dummy_index=True    #Dummy index means that the first element of each repeat is unused, so we don't reorder it. default true
        )

        Shuffled_Alpha_Heterozygosity.append(
            alpha_shuffled
        )

        # ---------------------------------------------------------
        # Shuffle beta heterozygosity
        #
        # Same permutation applied to rows AND columns of
        # every beta matrix at every time point and repeat
        # ---------------------------------------------------------

        beta_shuffled = []

        for repeat in beta_heterozygosity_matrices_list[i]:

            beta_shuffled.append(
                reorder_beta_heterozygosity(
                    repeat,
                    order
                )
            )

        Shuffled_Beta_Heterozygosity.append(
            beta_shuffled
        )


#=============================================================================#
# SAVE SHUFFLED DATA
#=============================================================================#

results_path = os.path.join(
    str(args.d),
    "Analysis_Results_Shuffled.npz"
)

np.savez(
    results_path,

    matrixlist=np.asarray(
        Shuffled_MatrixList,
        dtype=object
    ),

    Exponential_Decay_parameters=np.asarray(
        Exponential_Decay_parameters_shuffle,
        dtype=object
    ),

    alpha_heterozygosity_curves_list=np.asarray(
        Shuffled_Alpha_Heterozygosity,
        dtype=object
    ),

    beta_heterozygosity_matrices_list=np.asarray(
        Shuffled_Beta_Heterozygosity,
        dtype=object
    )
)
"""
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
"""
endtime = time.time()

print("Time taken:",endtime-starttime)
