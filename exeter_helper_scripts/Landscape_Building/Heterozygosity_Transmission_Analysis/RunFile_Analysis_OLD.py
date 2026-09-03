import numpy as np
import matplotlib.pyplot as plt

import Heterozygosity_ai
import Matrix_Analysis

import time

import subprocess
import os, shutil
import argparse
import re

starttime = time.time()

#=============================================================================#
# ARGPARSER
#=============================================================================#
parser = argparse.ArgumentParser(
    prog="Heterozygosity_Run.py",
    description="Run heterozygosity analysis over all landscape repeats"
)

parser.add_argument(
    "-d",
    type=str,
    required=True,
    help="Directory containing all the landscape repeats"
)

args = parser.parse_args()


#=============================================================================#
# FIND REPEAT DIRECTORIES
#=============================================================================#

SaveDir = args.d

repeat_dirs = []

for dirname in os.listdir(SaveDir):

    full_path = os.path.join(SaveDir, dirname)

    if not os.path.isdir(full_path):
        continue

    match = re.fullmatch(r"Repeat_(\d+)", dirname)

    if match:
        repeat_number = int(match.group(1))
        repeat_dirs.append((repeat_number, full_path))

repeat_dirs.sort(key=lambda x: x[0])

print("Found", len(repeat_dirs), "repeats")


#=============================================================================#
# ANALYSE EACH REPEAT
#=============================================================================#

Exponential_Decay_parameters = []
statslist = []

for repeat_number, repeat_dir in repeat_dirs:

    print("\n" + "=" * 70)
    print("Processing Repeat", repeat_number)
    print("=" * 70)


    #========================================================================#
    # FIND CDMATRIX FOR THIS REPEAT
    #========================================================================#

    cdmatrix_path = os.path.join(
        repeat_dir,
        "inputs",
        "cdmats",
        "cdmatrix.csv"
    )

    if not os.path.isfile(cdmatrix_path):

        print(
            f"WARNING: cdmatrix.csv does not exist for "
            f"Repeat_{repeat_number}: {cdmatrix_path}"
        )

        continue

    print("CD matrix:")
    print(cdmatrix_path)

    # Placeholder for future matrix analysis
    statslist.append(Matrix_Analysis.main(cdmatrix_path))


    #========================================================================#
    # FIND RANDOM-NUMBER DIRECTORY FOR HETEROZYGOSITY ANALYSIS
    #========================================================================#

    raw_dir = os.path.join(
        repeat_dir,
        "outputs",
        "raw"
    )

    if not os.path.isdir(raw_dir):

        print(
            f"WARNING: raw directory does not exist for "
            f"Repeat_{repeat_number}: {raw_dir}"
        )

        continue

    raw_subdirs = [
        dirname
        for dirname in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, dirname))
    ]

    if len(raw_subdirs) == 0:

        print(
            f"WARNING: No directory found inside raw for "
            f"Repeat_{repeat_number}"
        )

        continue

    if len(raw_subdirs) > 1:

        print(
            f"WARNING: Multiple directories found inside raw for "
            f"Repeat_{repeat_number}: {raw_subdirs}"
        )

        print("Skipping this repeat.")

        continue

    random_dir = raw_subdirs[0]

    analysis_dir = os.path.join(
        raw_dir,
        random_dir
    )

    print("Analysis directory:")
    print(analysis_dir)


    #========================================================================#
    # RUN HETEROZYGOSITY ANALYSIS
    #========================================================================#

    Exponential_Decay_parameters.append(
        Heterozygosity_ai.main(analysis_dir)
    )


#=============================================================================#
# FINISHED
#============================================================================#

print(Exponential_Decay_parameters)

endtime = time.time()

print("\n" + "=" * 70)
print("All repeats processed")
print("Time taken:", endtime - starttime, "seconds")
print("=" * 70)



#Plotting
Mixing_Times = [x["mixing_time"] for x in statslist]

plt.loglog(Mixing_Times, Exponential_Decay_parameters, 'o')
plt.xlabel("Mixing Time")
plt.ylabel("Heterozygosity Decay Parameter (a)")

plt.savefig(str(SaveDir) + "/Exponential_Decay_vs_Mixing_Time.png")
#plt.show()
plt.close()


Kemeny_Constants = [x["kemeny_constant"] for x in statslist]

plt.loglog(Kemeny_Constants, Exponential_Decay_parameters, 'o')
plt.xlabel("Kemeny_Constant")
plt.ylabel("Heterozygosity Decay Parameter (a)")

plt.savefig(str(SaveDir) + "/Exponential_Decay_vs_Kemeny_Constant.png")
#plt.show()
plt.close()

Conductances = [x["conductance"] for x in statslist]

plt.loglog(Conductances, Exponential_Decay_parameters, 'o')
plt.xlabel("Conductance")
plt.ylabel("Heterozygosity Decay Parameter (a)")

plt.savefig(str(SaveDir) + "/Exponential_Decay_vs_Conductance.png")
#plt.show()
plt.close()












#=============================================================================#
# PCA OF LANDSCAPE STRUCTURE
#
# The PCA is performed ONLY on landscape/network properties.
#
# Heterozygosity decay is treated as the response variable.
#
# We separate:
#
#     1. GLOBAL structure
#     2. LOCAL structure
#
# This allows us to ask whether global and local landscape structure
# independently explain heterozygosity decay.
#=============================================================================#

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np


#=============================================================================#
# BUILD DATAFRAME
#=============================================================================#

Landscape_Data = pd.DataFrame({

    # Global structure
    "Mixing Time":
        [x["mixing_time"] for x in statslist],

    "Kemeny Constant":
        [x["kemeny_constant"] for x in statslist],

    "Conductance":
        [x["conductance"] for x in statslist],

    # Local structure
    "Mean Effective Degree":
        [x["mean_effective_degree"] for x in statslist],

    "CV Effective Degree":
        [x["cv_effective_degree"] for x in statslist],

    "CV Stationary Distribution":
        [x["cv_stationary_distribution"] for x in statslist],

    "Mean Clustering":
        [x["mean_clustering"] for x in statslist]
})


# Add heterozygosity decay ONLY as a response variable
Landscape_Data["Heterozygosity Decay"] = (
    Exponential_Decay_parameters
)


# Remove invalid rows
Landscape_Data = Landscape_Data.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


print("\n================ LANDSCAPE DATA =================")

print(
    Landscape_Data.describe()
)


#=============================================================================#
# GLOBAL PCA
#=============================================================================#

Global_Variables = [
    "Mixing Time",
    "Kemeny Constant",
    "Conductance"
]

Global_Data = Landscape_Data[
    Global_Variables
].copy()


#-----------------------------------------------------------------------------#
# Log transform
#
# These three quantities are positive and can span substantial ranges.
#-----------------------------------------------------------------------------#

Global_Data_Log = np.log10(
    Global_Data
)


#-----------------------------------------------------------------------------#
# Standardise
#-----------------------------------------------------------------------------#

Global_Scaler = StandardScaler()

Global_Scaled = Global_Scaler.fit_transform(
    Global_Data_Log
)


#-----------------------------------------------------------------------------#
# PCA
#-----------------------------------------------------------------------------#

Global_PCA = PCA()

Global_PCA_Results = Global_PCA.fit_transform(
    Global_Scaled
)


#-----------------------------------------------------------------------------#
# Print explained variance
#-----------------------------------------------------------------------------#

print("\n================ GLOBAL PCA =================")

print("\nExplained variance:")

for i, variance in enumerate(
        Global_PCA.explained_variance_ratio_):

    print(
        f"PC{i+1}: {variance:.3f} "
        f"({100 * variance:.1f}%)"
    )


#-----------------------------------------------------------------------------#
# Loadings
#-----------------------------------------------------------------------------#

Global_Loadings = pd.DataFrame(

    Global_PCA.components_.T,

    index=Global_Variables,

    columns=[
        f"PC{i+1}"
        for i in range(Global_PCA.n_components_)
    ]
)

print("\nGlobal PCA loadings:")
print(Global_Loadings)

Global_Loadings.to_csv(
    os.path.join(
        SaveDir,
        "Global_PCA_Loadings.csv"
    )
)


#-----------------------------------------------------------------------------#
# Variable correlations with PCs
#-----------------------------------------------------------------------------#

Global_Correlations = pd.DataFrame(

    Global_PCA.components_.T *
    np.sqrt(
        Global_PCA.explained_variance_
    ),

    index=Global_Variables,

    columns=[
        f"PC{i+1}"
        for i in range(Global_PCA.n_components_)
    ]
)

print("\nGlobal variable correlations with PCs:")
print(Global_Correlations)

Global_Correlations.to_csv(
    os.path.join(
        SaveDir,
        "Global_PCA_Variable_Correlations.csv"
    )
)


#-----------------------------------------------------------------------------#
# Extract Global PC1
#-----------------------------------------------------------------------------#

Global_PC1 = Global_PCA_Results[:, 0]


#=============================================================================#
# LOCAL PCA
#=============================================================================#

Local_Variables = [

    "Mean Effective Degree",
    "CV Effective Degree",
    "CV Stationary Distribution",
    "Mean Clustering"
]

Local_Data = Landscape_Data[
    Local_Variables
].copy()


#-----------------------------------------------------------------------------#
# Standardise
#
# Unlike the global variables, we do NOT log-transform these quantities.
# They are already dimensionless structural statistics.
#-----------------------------------------------------------------------------#

Local_Scaler = StandardScaler()

Local_Scaled = Local_Scaler.fit_transform(
    Local_Data
)


#-----------------------------------------------------------------------------#
# PCA
#-----------------------------------------------------------------------------#

Local_PCA = PCA()

Local_PCA_Results = Local_PCA.fit_transform(
    Local_Scaled
)


#-----------------------------------------------------------------------------#
# Print explained variance
#-----------------------------------------------------------------------------#

print("\n================ LOCAL PCA =================")

print("\nExplained variance:")

for i, variance in enumerate(
        Local_PCA.explained_variance_ratio_):

    print(
        f"PC{i+1}: {variance:.3f} "
        f"({100 * variance:.1f}%)"
    )


#-----------------------------------------------------------------------------#
# Loadings
#-----------------------------------------------------------------------------#

Local_Loadings = pd.DataFrame(

    Local_PCA.components_.T,

    index=Local_Variables,

    columns=[
        f"PC{i+1}"
        for i in range(Local_PCA.n_components_)
    ]
)

print("\nLocal PCA loadings:")
print(Local_Loadings)

Local_Loadings.to_csv(
    os.path.join(
        SaveDir,
        "Local_PCA_Loadings.csv"
    )
)


#-----------------------------------------------------------------------------#
# Variable correlations with PCs
#-----------------------------------------------------------------------------#

Local_Correlations = pd.DataFrame(

    Local_PCA.components_.T *
    np.sqrt(
        Local_PCA.explained_variance_
    ),

    index=Local_Variables,

    columns=[
        f"PC{i+1}"
        for i in range(Local_PCA.n_components_)
    ]
)

print("\nLocal variable correlations with PCs:")
print(Local_Correlations)

Local_Correlations.to_csv(
    os.path.join(
        SaveDir,
        "Local_PCA_Variable_Correlations.csv"
    )
)


#-----------------------------------------------------------------------------#
# Extract Local PC1
#-----------------------------------------------------------------------------#

Local_PC1 = Local_PCA_Results[:, 0]


#=============================================================================#
# HETEROZYGOSITY DECAY
#=============================================================================#

Decay = Landscape_Data[
    "Heterozygosity Decay"
].values

Decay_Log = np.log10(
    Decay
)


#=============================================================================#
# GLOBAL PC1 VS HETEROZYGOSITY DECAY
#=============================================================================#

Global_Pearson_r, Global_Pearson_p = pearsonr(
    Global_PC1,
    Decay_Log
)

Global_Spearman_rho, Global_Spearman_p = spearmanr(
    Global_PC1,
    Decay_Log
)


print(
    "\n================ GLOBAL STRUCTURE VS DECAY ================"
)

print(
    f"Pearson r  = {Global_Pearson_r:.4f}"
)

print(
    f"Pearson p  = {Global_Pearson_p:.4e}"
)

print(
    f"Spearman rho = {Global_Spearman_rho:.4f}"
)

print(
    f"Spearman p   = {Global_Spearman_p:.4e}"
)


#-----------------------------------------------------------------------------#
# Plot
#-----------------------------------------------------------------------------#

plt.figure(figsize=(8, 6))

plt.scatter(
    Global_PC1,
    Decay_Log,
    alpha=0.7
)

Slope, Intercept = np.polyfit(
    Global_PC1,
    Decay_Log,
    1
)

X_Fit = np.linspace(
    Global_PC1.min(),
    Global_PC1.max(),
    100
)

Y_Fit = (
    Slope * X_Fit +
    Intercept
)

plt.plot(
    X_Fit,
    Y_Fit
)

plt.xlabel("Global Structure PC1")

plt.ylabel(
    "log10(Heterozygosity Decay Parameter)"
)

plt.title(
    "Global Landscape Structure vs Heterozygosity Decay"
    f"\nr = {Global_Pearson_r:.3f}"
)

plt.savefig(
    os.path.join(
        SaveDir,
        "Global_PC1_vs_Heterozygosity_Decay.png"
    ),
    bbox_inches="tight"
)

plt.close()


#=============================================================================#
# LOCAL PC1 VS HETEROZYGOSITY DECAY
#=============================================================================#

Local_Pearson_r, Local_Pearson_p = pearsonr(
    Local_PC1,
    Decay_Log
)

Local_Spearman_rho, Local_Spearman_p = spearmanr(
    Local_PC1,
    Decay_Log
)


print(
    "\n================ LOCAL STRUCTURE VS DECAY ================"
)

print(
    f"Pearson r  = {Local_Pearson_r:.4f}"
)

print(
    f"Pearson p  = {Local_Pearson_p:.4e}"
)

print(
    f"Spearman rho = {Local_Spearman_rho:.4f}"
)

print(
    f"Spearman p   = {Local_Spearman_p:.4e}"
)


#-----------------------------------------------------------------------------#
# Plot
#-----------------------------------------------------------------------------#

plt.figure(figsize=(8, 6))

plt.scatter(
    Local_PC1,
    Decay_Log,
    alpha=0.7
)

Slope, Intercept = np.polyfit(
    Local_PC1,
    Decay_Log,
    1
)

X_Fit = np.linspace(
    Local_PC1.min(),
    Local_PC1.max(),
    100
)

Y_Fit = (
    Slope * X_Fit +
    Intercept
)

plt.plot(
    X_Fit,
    Y_Fit
)

plt.xlabel("Local Structure PC1")

plt.ylabel(
    "log10(Heterozygosity Decay Parameter)"
)

plt.title(
    "Local Landscape Structure vs Heterozygosity Decay"
    f"\nr = {Local_Pearson_r:.3f}"
)

plt.savefig(
    os.path.join(
        SaveDir,
        "Local_PC1_vs_Heterozygosity_Decay.png"
    ),
    bbox_inches="tight"
)

plt.close()


#=============================================================================#
# EXPLAINED VARIANCE PLOTS
#=============================================================================#

plt.figure()

plt.plot(
    range(
        1,
        Global_PCA.n_components_ + 1
    ),
    Global_PCA.explained_variance_ratio_,
    'o-'
)

plt.xlabel("Principal Component")

plt.ylabel(
    "Proportion of Variance Explained"
)

plt.title("Global Landscape PCA")

plt.savefig(
    os.path.join(
        SaveDir,
        "Global_PCA_Explained_Variance.png"
    ),
    bbox_inches="tight"
)

plt.close()


plt.figure()

plt.plot(
    range(
        1,
        Local_PCA.n_components_ + 1
    ),
    Local_PCA.explained_variance_ratio_,
    'o-'
)

plt.xlabel("Principal Component")

plt.ylabel(
    "Proportion of Variance Explained"
)

plt.title("Local Landscape PCA")

plt.savefig(
    os.path.join(
        SaveDir,
        "Local_PCA_Explained_Variance.png"
    ),
    bbox_inches="tight"
)

plt.close()


#=============================================================================#
# GLOBAL PCA BIPLOT
#=============================================================================#

plt.figure(figsize=(8, 8))

plt.scatter(
    Global_PCA_Results[:, 0],
    Global_PCA_Results[:, 1],
    alpha=0.7
)

Scale = 2.5

for i, variable in enumerate(Global_Variables):

    x = (
        Global_PCA.components_[0, i] *
        Scale
    )

    y = (
        Global_PCA.components_[1, i] *
        Scale
    )

    plt.arrow(
        0,
        0,
        x,
        y,
        head_width=0.05,
        length_includes_head=True
    )

    plt.text(
        x * 1.1,
        y * 1.1,
        variable,
        ha="center",
        va="center"
    )

plt.axhline(
    0,
    linewidth=0.5
)

plt.axvline(
    0,
    linewidth=0.5
)

plt.xlabel(
    f"Global PC1 "
    f"({100 * Global_PCA.explained_variance_ratio_[0]:.1f}%)"
)

plt.ylabel(
    f"Global PC2 "
    f"({100 * Global_PCA.explained_variance_ratio_[1]:.1f}%)"
)

plt.title(
    "PCA of Global Landscape Structure"
)

plt.savefig(
    os.path.join(
        SaveDir,
        "Global_PCA_Biplot.png"
    ),
    bbox_inches="tight"
)

plt.close()


#=============================================================================#
# LOCAL PCA BIPLOT
#=============================================================================#

plt.figure(figsize=(8, 8))

plt.scatter(
    Local_PCA_Results[:, 0],
    Local_PCA_Results[:, 1],
    alpha=0.7
)

Scale = 2.5

for i, variable in enumerate(Local_Variables):

    x = (
        Local_PCA.components_[0, i] *
        Scale
    )

    y = (
        Local_PCA.components_[1, i] *
        Scale
    )

    plt.arrow(
        0,
        0,
        x,
        y,
        head_width=0.05,
        length_includes_head=True
    )

    plt.text(
        x * 1.1,
        y * 1.1,
        variable,
        ha="center",
        va="center"
    )

plt.axhline(
    0,
    linewidth=0.5
)

plt.axvline(
    0,
    linewidth=0.5
)

plt.xlabel(
    f"Local PC1 "
    f"({100 * Local_PCA.explained_variance_ratio_[0]:.1f}%)"
)

plt.ylabel(
    f"Local PC2 "
    f"({100 * Local_PCA.explained_variance_ratio_[1]:.1f}%)"
)

plt.title(
    "PCA of Local Landscape Structure"
)

plt.savefig(
    os.path.join(
        SaveDir,
        "Local_PCA_Biplot.png"
    ),
    bbox_inches="tight"
)

plt.close()


#=============================================================================#
# SUMMARY
#=============================================================================#

print(
    "\n================ LANDSCAPE PCA SUMMARY ================"
)

print(
    f"\nGlobal PC1 explains "
    f"{100 * Global_PCA.explained_variance_ratio_[0]:.1f}% "
    f"of global structural variation."
)

print(
    f"Correlation with decay: "
    f"r = {Global_Pearson_r:.4f}"
)

print(
    f"\nLocal PC1 explains "
    f"{100 * Local_PCA.explained_variance_ratio_[0]:.1f}% "
    f"of local structural variation."
)

print(
    f"Correlation with decay: "
    f"r = {Local_Pearson_r:.4f}"
)

print(
    "\nPCA results saved to:"
)

print(
    str(SaveDir)
)