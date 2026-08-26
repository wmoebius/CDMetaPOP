"""
This script exists for the purpose of calculating key statistics of markov chain matrices. 
You can either specify a cdmat file and explore their statistics, or specify parameters for repeated construction and averages.
"""

import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt




# ========================
# FUNCTIONS
# ========================
def Mixing_Time(Dispersal_Matrix):
    eigenvalues = abs(np.linalg.eigvals(Dispersal_Matrix))  # Use abs() if looking for magnitude
    
    # Sort indices in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # Get the second largest eigenvalue
    second_largest = eigenvalues[sorted_indices[1]]

    print("|Eigenvalues|:",eigenvalues)
    print("Second largest eigenvalue:", second_largest)
    print("Mixing time:",1/(1-second_largest))

    return 1/(1-second_largest)



import numpy as np


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
        raise ValueError("ProbMatrix must be a square matrix.")

    n = P.shape[0]

    # ---------------------------------------------------------
    # Check that matrix is approximately row stochastic
    # ---------------------------------------------------------

    row_sums = P.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError(
            "ProbMatrix must be row-stochastic: "
            "each row must sum to 1."
        )

    # ---------------------------------------------------------
    # 1. Incoming strength
    #
    # s_i = sum_j P[j, i]
    # ---------------------------------------------------------

    incoming_strength = P.sum(axis=0)

    # Normalised incoming strength
    q = incoming_strength / incoming_strength.sum()

    # ---------------------------------------------------------
    # 2. Hub concentration
    #
    # C = sum_i q_i^2
    # ---------------------------------------------------------

    hub_concentration = np.sum(q**2)

    # Effective number of hubs
    N_hub = 1.0 / hub_concentration

    # ---------------------------------------------------------
    # 3. Strongest hub
    # ---------------------------------------------------------

    sorted_q = np.sort(q)[::-1]

    strongest_hub = sorted_q[0]

    # Cumulative contribution of top k hubs
    cumulative_hub_contribution = np.cumsum(sorted_q)

    # ---------------------------------------------------------
    # 4. Stationary distribution
    #
    # Find pi such that
    #
    # pi P = pi
    #
    # using the eigenvector of P.T with eigenvalue 1.
    # ---------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))

    pi = np.real(eigenvectors[:, idx])

    # Eigenvectors can have arbitrary sign
    if np.sum(pi) < 0:
        pi = -pi

    # Normalise
    pi = pi / np.sum(pi)

    # Remove tiny numerical errors
    pi = np.maximum(pi, 0)
    pi = pi / np.sum(pi)

    # ---------------------------------------------------------
    # 5. Effective number of patches in stationary distribution
    # ---------------------------------------------------------

    stationary_concentration = np.sum(pi**2)

    N_eff = 1.0 / stationary_concentration

    # ---------------------------------------------------------
    # Return results
    # ---------------------------------------------------------

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












# =======================
# ARGUMENTS
# =======================

parser = argparse.ArgumentParser(
    description = "Matrix Analysis"
)

parser.add_argument(
    "-d",
    type = str,
    default = "False",
    help = 'The CDMAT.csv'
)


    
parser.add_argument("-n",
                    type=int,
                    default=10,
                    help='Number of patches to create')

def DistChoices(value):
    if value in ["Linear","Exponential","Gaussian","Power"]:
        return value
    else:
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value for -ProbDist: {value}. Must be 'Linear', 'Exponential', 'Gaussian', 'Power', or a float.")

        if value <= 0:
            raise argparse.ArgumentTypeError(f"Invalid value for -ProbDist: {value}. Must be a positive float.")
        return value


parser.add_argument("-ProbDist",
                    type=DistChoices,
                    default = "0.1",
                    help='Probability distribution to use for dispersal: Linear, Exponential, Gaussian, Power, or a float value for radius of connection.')


parser.add_argument("-param1",
                    type=float,
                    default = 1,
                    help = "Parameter 1 for the probability distribution (if applicable)")

parser.add_argument("-param2",
                    type=float,
                    default = 1,
                    help = "Parameter 2 for the probability distribution (if applicable)")

parser.add_argument("-repeats",
                    type = int,
                    default = 10)

args = parser.parse_args()


if args.d != "False":

    # =======================
    # CHECK FILE
    # =======================
    cdmat_file = Path(args.d)

    if not cdmat_file.is_file():
        raise FileNotFoundError(
            f"Not a file: {cdmat_file}"
        )


    # ========================
    # Exctract Matrix
    # =========================

    Dispersal_Matrix = np.genfromtxt(cdmat_file,delimiter=",")

    # =========================
    # Find the statistics
    # =========================
    # Compute the mixing times
    Mixing_Time(Dispersal_Matrix)

    #Hub  statistics
    stats = calculate_hub_statistics(Dispersal_Matrix)

    print("Effective number of hubs:",
        stats["effective_number_hubs"])

    print("Strongest hub fraction:",
        stats["strongest_hub_fraction"])

    print("Effective number of patches:",
        stats["effective_number_patches"])

else:

    np.random.seed(0)

    #Data:
    Eigenval_second = []

    for r in range(args.repeats):
        print(r)
        #Generate landscape:
        Locations = []
        
        for i in range(args.n):
            x = np.random.random()#*scale_x
            y = np.random.random()#*scale_y
            
            Locations.append((x,y))
            
        Locations = np.asarray(Locations)
        #print("Locations:",Locations)

        """
        #If sticky_radius is greater than 0, find connected components and combine them into single nodes.
        if args.sticky_radius > 0:
            Components = Find_Connected_Components(Locations, args.sticky_radius)
    
    
            New_locations = []
            New_locations_Sizes = []
    
            for i in Components:
                x = 0
                y = 0
                for j in i:
                    x+= j[0]
                    y+= j[1]
                    
                New_locations.append((x/len(i),y/len(i)))
                New_locations_Sizes.append(len(i))
                
            Locations = np.asarray(New_locations)
            Locations_Sizes = np.asarray(New_locations_Sizes)
        """
    
    
    
        DistMatrix = np.zeros(shape=(len(Locations),len(Locations)))
    
        for i in range(len(DistMatrix)):
            for j in range(i,len(DistMatrix)):
                
                
                distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                                (Locations[i][1]-Locations[j][1])**2)
                
                DistMatrix[i][j] = DistMatrix[j][i] = distance


        if type(args.ProbDist) == float:
            radius = args.ProbDist
            ProbMatrix = (DistMatrix <= radius).astype(float)
    
            
    
        elif args.ProbDist == "Linear":
            ProbMatrix = 1 - (DistMatrix / DistMatrix.max())
    
        elif args.ProbDist == "Exponential":
            ProbMatrix = np.exp(-args.param1 * DistMatrix)
    
        elif args.ProbDist == "Gaussian":
            sigma = DistMatrix.mean()
            ProbMatrix = np.exp(-DistMatrix**2 / (2 * sigma**2))
    
        elif args.ProbDist == "Power":
            ProbMatrix = 1 / (DistMatrix**args.param1)
            ProbMatrix[DistMatrix == 0] = 0  # Avoid division by zero
    
    
        #Normalise
        np.fill_diagonal(ProbMatrix, 0)
        # Row sums
        row_sums = ProbMatrix.sum(axis=1, keepdims=True)


        # Only normalise rows with at least one possible destination
        non_empty = row_sums[:, 0] > 0

        ProbMatrix[non_empty] /= row_sums[non_empty]

        #If all zero for a row, make that connect to itself.
        if type(args.ProbDist) == float:
            zero_rows = np.sum(ProbMatrix, axis=1) == 0
            ProbMatrix[zero_rows, np.arange(ProbMatrix.shape[0])[zero_rows]] = 1


        #Eigenvalues
        """
        eigenvalues = abs(np.linalg.eigvals(ProbMatrix))  # Use abs() if looking for magnitude
        
        # Sort indices in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
    
        # Get the second largest eigenvalue
        second_largest = eigenvalues[sorted_indices[1]]
    
        #print("|Eigenvalues|:",eigenvalues)
        #print("Second largest eigenvalue:", second_largest)
        #print("Mixing time:",1/(1-second_largest))
        """
        Eigenval_second.append(1-1/Mixing_Time(ProbMatrix))

    print(Eigenval_second)
    plt.hist(Eigenval_second,bins=100)
    plt.title("Mean = %0.3f, Median = %0.3f"%(np.mean(Eigenval_second),
                                              np.median(Eigenval_second)))
    plt.savefig("MatrixAnalysis_n%d_ProbDist_%s_Param_%s_Repeats_%d.png"%(
        args.n,args.ProbDist,args.param1,args.repeats)
    )
    plt.show()
    plt.close()

    Mixing_Time = 1/(1-np.asarray(Eigenval_second))
    log_bins = np.logspace(0, np.log10(Mixing_Time.max()), 100)

    plt.hist(Mixing_Time,bins=log_bins)
    plt.title("Mean = %0.3f, Median = %0.3f"%(np.mean(Mixing_Time),
                                              np.median(Mixing_Time)))

    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("MatrixAnalysis_n%d_ProbDist_%s_Param_%s_Repeats_%d_MIXING.png"%(
        args.n,args.ProbDist,args.param1,args.repeats)
    )
    plt.show()


