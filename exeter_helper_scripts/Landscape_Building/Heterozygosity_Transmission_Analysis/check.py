import numpy as np
import os
import argparse


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
