import numpy as np
import matplotlib.pyplot as plt
import Heterozygosity_ai
import Matrix_Analysis
import time, subprocess, os, shutil, argparse, re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

starttime = time.time()

#=============================================================================#
# ARGPARSER
#=============================================================================#
parser = argparse.ArgumentParser(prog="Heterozygosity_Run.py", description="Run heterozygosity analysis over all landscape repeats")
parser.add_argument("-d", type=str, required=True, help="Directory containing all the landscape repeats")
args = parser.parse_args()
SaveDir = args.d

#=============================================================================#
# FIND REPEAT DIRECTORIES
#=============================================================================#
repeat_dirs = []
for dirname in os.listdir(SaveDir):
    full_path = os.path.join(SaveDir, dirname)
    if not os.path.isdir(full_path): continue
    match = re.fullmatch(r"Repeat_(\d+)", dirname)
    if match: repeat_dirs.append((int(match.group(1)), full_path))
repeat_dirs.sort(key=lambda x: x[0])
print("Found", len(repeat_dirs), "repeats")

#=============================================================================#
# ANALYSE EACH REPEAT
#=============================================================================#
Exponential_Decay_parameters, statslist = [], []
for repeat_number, repeat_dir in repeat_dirs:
    print("\n" + "=" * 70); print("Processing Repeat", repeat_number); print("=" * 70)
    cdmatrix_path = os.path.join(repeat_dir, "inputs", "cdmats", "cdmatrix.csv")
    if not os.path.isfile(cdmatrix_path):
        print(f"WARNING: cdmatrix.csv does not exist for Repeat_{repeat_number}: {cdmatrix_path}"); continue
    print("CD matrix:", cdmatrix_path)
    statslist.append(Matrix_Analysis.main(cdmatrix_path))

    raw_dir = os.path.join(repeat_dir, "outputs", "raw")
    if not os.path.isdir(raw_dir):
        print(f"WARNING: raw directory does not exist for Repeat_{repeat_number}: {raw_dir}"); continue
    raw_subdirs = [dirname for dirname in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, dirname))]
    if len(raw_subdirs) == 0:
        print(f"WARNING: No directory found inside raw for Repeat_{repeat_number}"); continue
    if len(raw_subdirs) > 1:
        print(f"WARNING: Multiple directories found inside raw for Repeat_{repeat_number}: {raw_subdirs}"); print("Skipping this repeat."); continue
    analysis_dir = os.path.join(raw_dir, raw_subdirs[0])
    print("Analysis directory:", analysis_dir)
    Exponential_Decay_parameters.append(Heterozygosity_ai.main(analysis_dir))

#=============================================================================#
# FINISHED
#=============================================================================#
print(Exponential_Decay_parameters)
endtime = time.time()
print("\n" + "=" * 70); print("All repeats processed"); print("Time taken:", endtime - starttime, "seconds"); print("=" * 70)

#=============================================================================#
# BASIC PLOTS
#=============================================================================#
Mixing_Times = [x["mixing_time"] for x in statslist]
plt.loglog(Mixing_Times, Exponential_Decay_parameters, 'o'); plt.xlabel("Mixing Time"); plt.ylabel("Heterozygosity Decay Parameter (a)"); plt.savefig(os.path.join(SaveDir, "Exponential_Decay_vs_Mixing_Time.png")); plt.close()

Kemeny_Constants = [x["kemeny_constant"] for x in statslist]
plt.loglog(Kemeny_Constants, Exponential_Decay_parameters, 'o'); plt.xlabel("Kemeny_Constant"); plt.ylabel("Heterozygosity Decay Parameter (a)"); plt.savefig(os.path.join(SaveDir, "Exponential_Decay_vs_Kemeny_Constant.png")); plt.close()

Conductances = [x["conductance"] for x in statslist]
plt.loglog(Conductances, Exponential_Decay_parameters, 'o'); plt.xlabel("Conductance"); plt.ylabel("Heterozygosity Decay Parameter (a)"); plt.savefig(os.path.join(SaveDir, "Exponential_Decay_vs_Conductance.png")); plt.close()

#=============================================================================#
# CORRELATION, PCA AND REGRESSION
#=============================================================================#
# The response (heterozygosity decay) is deliberately excluded from the PCA.
# PCA should describe the structure of the PARAMETERS, not optimise directions
# according to the response. We therefore analyse each parameter group in three
# separate steps: (1) correlate each parameter directly with the response,
# (2) perform PCA using only the parameters, and (3) regress the response on ALL
# resulting PCs. The correlations answer which individual parameters are associated
# with decay; PCA identifies the major directions of parameter variation; regression
# then determines whether those parameter-space directions explain variation in decay.
# Global parameters are log10-transformed before PCA/correlation because they are
# positive and span substantial ranges. Local parameters are not log-transformed.
# Standardisation puts parameters on comparable scales before PCA; it does not change
# Pearson/Spearman correlations. The response is log10-transformed because decay is
# positive and the existing analysis treats its relationship on a logarithmic scale.
# Using ALL PCs in the regression retains all information in the parameter set:
# PCA merely rotates the standardised parameter axes into orthogonal coordinates.

Landscape_Data = pd.DataFrame({
    "Mixing Time": [x["mixing_time"] for x in statslist],
    "Kemeny Constant": [x["kemeny_constant"] for x in statslist],
    "Conductance": [x["conductance"] for x in statslist],
    "Mean Effective Degree": [x["mean_effective_degree"] for x in statslist],
    "CV Effective Degree": [x["cv_effective_degree"] for x in statslist],
    "CV Stationary Distribution": [x["cv_stationary_distribution"] for x in statslist],
    "Mean Clustering": [x["mean_clustering"] for x in statslist],
    "Heterozygosity Decay": Exponential_Decay_parameters
}).replace([np.inf, -np.inf], np.nan).dropna()
print("\n================ LANDSCAPE DATA ================="); print(Landscape_Data.describe())

Global_Variables = ["Mixing Time", "Kemeny Constant", "Conductance"]
Local_Variables = ["Mean Effective Degree", "CV Effective Degree", "CV Stationary Distribution", "Mean Clustering"]
Decay = Landscape_Data["Heterozygosity Decay"].values
Decay_Log = np.log10(Decay)

def analyse_group(data, variables, name, log_transform=False):
    # Correlation uses the same parameter transformation that enters the PCA.
    # Pearson measures linear association; Spearman measures monotonic association.
    # We calculate both against log10(decay), giving a direct measure of how each
    # individual parameter relates to the response before PCA combines parameters.
    transformed = np.log10(data[variables]) if log_transform else data[variables].copy()
    scaled = StandardScaler().fit_transform(transformed)
    Correlations = pd.DataFrame(index=variables, columns=["Pearson r", "Pearson p", "Spearman rho", "Spearman p"], dtype=float)
    for variable in variables:
        Correlations.loc[variable] = [*pearsonr(transformed[variable], Decay_Log), *spearmanr(transformed[variable], Decay_Log)]
    print(f"\n================ {name.upper()} PARAMETER CORRELATIONS ================="); print(Correlations)
    Correlations.to_csv(os.path.join(SaveDir, f"{name}_Parameter_Correlations.csv"))

    # PCA is performed ONLY on the standardised parameters, never on decay.
    # Each PC is therefore a direction in parameter space chosen to explain as much
    # parameter variance as possible. Its loadings show how strongly each original
    # parameter contributes to that direction. This is deliberately independent of y.
    pca = PCA().fit(scaled)
    PC_Results = pca.transform(scaled)
    PC_Names = [f"PC{i+1}" for i in range(pca.n_components_)]
    Loadings = pd.DataFrame(pca.components_.T, index=variables, columns=PC_Names)
    PC_Correlations = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=variables, columns=PC_Names)
    print(f"\n================ {name.upper()} PCA ================="); print("Explained variance:", " ".join(f"{pc}: {v:.3f} ({100*v:.1f}%)" for pc, v in zip(PC_Names, pca.explained_variance_ratio_))); print(f"\n{name} PCA loadings:"); print(Loadings)
    Loadings.to_csv(os.path.join(SaveDir, f"{name}_PCA_Loadings.csv")); PC_Correlations.to_csv(os.path.join(SaveDir, f"{name}_PCA_Variable_Correlations.csv"))

    # Regression now uses the PCs as predictors and log10(decay) as the response.
    # ALL PCs are included, so no information from the original parameter set is
    # discarded. Each coefficient measures the association between one orthogonal
    # parameter-space direction and decay while controlling for the other PCs.
    regression = LinearRegression().fit(PC_Results, Decay_Log)
    R2 = regression.score(PC_Results, Decay_Log)
    Regression = pd.DataFrame({"Coefficient": regression.coef_}, index=PC_Names)
    print(f"\n{name} PC regression:"); print(Regression); print(f"Intercept = {regression.intercept_:.4f}, R² = {R2:.4f}")
    Regression.to_csv(os.path.join(SaveDir, f"{name}_PC_Regression.csv"))

    # Retain a PC1-vs-response plot for visualisation; note that the actual regression above uses ALL PCs.
    PC1_r, PC1_p = pearsonr(PC_Results[:, 0], Decay_Log)
    PC1_rho, PC1_rho_p = spearmanr(PC_Results[:, 0], Decay_Log)
    plt.figure(figsize=(8, 6)); plt.scatter(PC_Results[:, 0], Decay_Log, alpha=0.7)
    slope, intercept = np.polyfit(PC_Results[:, 0], Decay_Log, 1); xfit = np.linspace(PC_Results[:, 0].min(), PC_Results[:, 0].max(), 100); plt.plot(xfit, slope*xfit + intercept)
    plt.xlabel(f"{name} Structure PC1"); plt.ylabel("log10(Heterozygosity Decay Parameter)"); plt.title(f"{name} Landscape Structure vs Heterozygosity Decay\nr = {PC1_r:.3f}"); plt.savefig(os.path.join(SaveDir, f"{name}_PC1_vs_Heterozygosity_Decay.png"), bbox_inches="tight"); plt.close()

    # Retain the explained-variance plot and PCA biplot from the original analysis.
    plt.figure(); plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, 'o-'); plt.xlabel("Principal Component"); plt.ylabel("Proportion of Variance Explained"); plt.title(f"{name} Landscape PCA"); plt.savefig(os.path.join(SaveDir, f"{name}_PCA_Explained_Variance.png"), bbox_inches="tight"); plt.close()
    if pca.n_components_ >= 2:
        plt.figure(figsize=(8, 8)); plt.scatter(PC_Results[:, 0], PC_Results[:, 1], alpha=0.7); Scale = 2.5
        for i, variable in enumerate(variables):
            x, y = pca.components_[0, i]*Scale, pca.components_[1, i]*Scale
            plt.arrow(0, 0, x, y, head_width=0.05, length_includes_head=True); plt.text(x*1.1, y*1.1, variable, ha="center", va="center")
        plt.axhline(0, linewidth=0.5); plt.axvline(0, linewidth=0.5); plt.xlabel(f"{name} PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"); plt.ylabel(f"{name} PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"); plt.title(f"PCA of {name} Landscape Structure"); plt.savefig(os.path.join(SaveDir, f"{name}_PCA_Biplot.png"), bbox_inches="tight"); plt.close()
    return pca, PC_Results, Correlations, Regression, R2, PC1_r, PC1_p, PC1_rho, PC1_rho_p

Global_PCA, Global_PCA_Results, Global_Correlations, Global_Regression, Global_R2, Global_Pearson_r, Global_Pearson_p, Global_Spearman_rho, Global_Spearman_p = analyse_group(Landscape_Data, Global_Variables, "Global", log_transform=True)
Local_PCA, Local_PCA_Results, Local_Correlations, Local_Regression, Local_R2, Local_Pearson_r, Local_Pearson_p, Local_Spearman_rho, Local_Spearman_p = analyse_group(Landscape_Data, Local_Variables, "Local", log_transform=False)

#=============================================================================#
# SUMMARY
#=============================================================================#
print("\n================ LANDSCAPE ANALYSIS SUMMARY ================")
print(f"Global PC1 explains {100*Global_PCA.explained_variance_ratio_[0]:.1f}% of global structural variation; PC regression R² = {Global_R2:.4f}")
print(f"Local PC1 explains {100*Local_PCA.explained_variance_ratio_[0]:.1f}% of local structural variation; PC regression R² = {Local_R2:.4f}")
print("\nAnalysis results saved to:", SaveDir)

