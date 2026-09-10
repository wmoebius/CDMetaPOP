import numpy as np
import os
import argparse
import copy

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


#=============================================================================#
# PRINT RESULTS
#=============================================================================#

print("\n" + "=" * 70)
print("matrixlist")
print("=" * 70)

print(matrixlist)


print("\n" + "=" * 70)
print("EXPONENTIAL DECAY PARAMETERS")
print("=" * 70)

print(Exponential_Decay_parameters)





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