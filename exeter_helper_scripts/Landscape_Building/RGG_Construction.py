# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:49:56 2026

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import argparse
import os
import networkx as nx

#=============================================================================#
# ARGPARSER
#=============================================================================#
parser = argparse.ArgumentParser(
    prog = "RGG_Construction.py",
    description="Create a landscape for CDmetaPOP")

parser.add_argument("-d",
                    type=str,
                    default='',
                    help='Directory to save data')

parser.add_argument("-i",
                    type=str,
                    default='',
                    help='Directory to copy ClassVars,PopVars, and RunVars')

parser.add_argument("-r",
                    default=0,
                    type=int,
                    help='New random seed: Zero creates a new random seed.')

parser.add_argument("-type",
                    type=str,
                    default='RGG',
                    help='Type of landscape to create: RGG, Lattice, or other...',
                    choices=['RGG','SLattice',"HLattice"])

parser.add_argument("-n",
                    type=int,
                    default=10,
                    help='Number of patches to create')

def DistChoices(value):
    if value in ["Linear","Exponential","Gaussian"]:
        return value
    else:
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value for -ProbDist: {value}. Must be 'Linear', 'Exponential', 'Gaussian', or a float.")

        if value <= 0:
            raise argparse.ArgumentTypeError(f"Invalid value for -ProbDist: {value}. Must be a positive float.")
        return value

parser.add_argument("-ProbDist",
                    type=DistChoices,
                    default = "1",
                    help='Probability distribution to use for dispersal: Linear, Exponential, Gaussian, or a float value for radius of connection.')

args = parser.parse_args()
print(args.d)

if args.r != 0:
    np.random.seed(args.r)


#Directory where the inputs are going to be placed.
if args.d != "":
    if not os.path.isdir(str(args.d)):
        os.makedirs(str(args.d))
        os.makedirs(str(args.d)+"/inputs/classvars")
        os.makedirs(str(args.d)+"/inputs/cdmats")
        os.makedirs(str(args.d)+"/inputs/patchvars")
        os.makedirs(str(args.d)+"/inputs/popvars")

        os.makedirs(str(args.d)+"/outputs/raw")
        os.makedirs(str(args.d)+"/outputs/analysed")


else:
    raise ValueError("Please specify a directory to save data using the -d flag.")

#Directory where the ClassVars, PopVars, and RunVars are located.
if args.i != "":
    if not os.path.isdir(str(args.i)):
        raise ValueError("The directory specified for the -i flag does not exist.")
    else:
        #Copy ClassVars, PopVars, and RunVars to the new directory.
        os.system("cp "+str(args.i)+"/classvars/ClassVars.csv "+str(args.d)+"/inputs/classvars/ClassVars.csv")
        os.system("cp "+str(args.i)+"/popvars/PopVars.csv "+str(args.d)+"/inputs/popvars/PopVars.csv")
        os.system("cp "+str(args.i)+"/RunVars.csv "+str(args.d)+"/inputs/RunVars.csv")


#=============================================================================#
# PARAMETERS
#=============================================================================#
#number of individual patches
n = args.n

#scale of the bounding box of patches
scale_x = 10
scale_y = 15

#=============================================================================#
# RANDOMLY CONSTRUCT LOCATIONS
#=============================================================================#
Locations = []

for i in range(n):
    x = np.random.random()*scale_x
    y = np.random.random()*scale_y
    
    Locations.append((x,y))
    
Locations = np.asarray(Locations)
print("Locations:",Locations)


#=============================================================================#
# CONSTUCT DISTANCE MATRIX
#=============================================================================#
if args.type == "RGG":
    DistMatrix = np.zeros(shape=(n,n))

    for i in range(len(DistMatrix)):
        for j in range(i,len(DistMatrix)):
            
            
            distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                            (Locations[i][1]-Locations[j][1])**2)
            
            DistMatrix[i][j] = DistMatrix[j][i] = distance
        
        
if args.type == "SLattice":
    #Create a square lattice of patches
    side_length = args.n
    x_coords = np.arange(side_length)
    y_coords = np.arange(side_length)
    
    Locations = np.array([(x, y) for x in x_coords for y in y_coords])
    print("Locations:",Locations)
    DistMatrix = np.zeros(shape=(len(Locations),len(Locations)))
    
    for i in range(len(DistMatrix)):
        for j in range(i,len(DistMatrix)):
            distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                            (Locations[i][1]-Locations[j][1])**2)
            
            DistMatrix[i][j] = DistMatrix[j][i] = distance

if args.type == "HLattice":
    #Create a hexagonal lattice of patches
    side_length = args.n
    x_coords = np.arange(side_length)
    y_coords = np.arange(side_length)
    
    Locations = []
    for i in range(side_length):
        for j in range(side_length):
            x = i + (j % 2) * 0.5  # Offset every other row
            y = j * (np.sqrt(3)/2)  # Vertical spacing for hexagonal grid
            Locations.append((x, y))
    
    Locations = np.array(Locations)
    print("Locations:",Locations)
    
    DistMatrix = np.zeros(shape=(len(Locations),len(Locations)))
    
    for i in range(len(DistMatrix)):
        for j in range(i,len(DistMatrix)):
            distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                            (Locations[i][1]-Locations[j][1])**2)
            
            DistMatrix[i][j] = DistMatrix[j][i] = distance
#=============================================================================#
# CONSTUCT Probability MATRIX
#=============================================================================#
#RGG example:


if args.type == "RGG":
    if type(args.ProbDist) == float:
        radius = args.ProbDist
        ProbMatrix = (DistMatrix <= radius).astype(float)

    elif args.ProbDist == "Linear":
        ProbMatrix = 1 - (DistMatrix / DistMatrix.max())

    elif args.ProbDist == "Exponential":
        ProbMatrix = np.exp(-DistMatrix)

    elif args.ProbDist == "Gaussian":
        sigma = DistMatrix.mean()
        ProbMatrix = np.exp(-DistMatrix**2 / (2 * sigma**2))


if args.type == "SLattice":
    ProbMatrix = (DistMatrix <= 1).astype(float)

if args.type == "HLattice":
    ProbMatrix = (DistMatrix <= 1.1).astype(float)

#Normalise
np.fill_diagonal(ProbMatrix, 0)
# Row sums
row_sums = ProbMatrix.sum(axis=1, keepdims=True)

# Only normalise rows with at least one possible destination
non_empty = row_sums[:, 0] > 0

ProbMatrix[non_empty] /= row_sums[non_empty]


#=============================================================================#
# PLOTTING
#=============================================================================#
# Create graph
G = nx.MultiDiGraph()

# Add nodes
for i, location in enumerate(Locations):
    G.add_node(i, pos=location)

# Add directed edges
for i in range(len(ProbMatrix)):
    for j in range(len(ProbMatrix)):
        if i != j and ProbMatrix[i, j] > 0:
            G.add_edge(
                i,
                j,
                weight=ProbMatrix[i, j]
            )

# Node positions
pos = {i: Locations[i] for i in range(len(Locations))}

fig, ax = plt.subplots(figsize=(10, 10))

# Draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=100,
    ax=ax,
    hide_ticks=False
)

# Draw edges
for u, v, key, data in G.edges(keys=True, data=True):

    probability = data["weight"]

    # Give the two directions opposite curvature
    if key == 0:
        rad = 0.25
    else:
        rad = -0.25

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v, key)],
        width=0.5 + 5 * probability,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        ax=ax,
        hide_ticks=False
    )

# Same axes setup as your scatter plot
ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_xlim([0, scale_x])
ax.set_ylim([0, scale_y])

if args.type == "SLattice":
    ax.set_xlim([0, args.n])
    ax.set_ylim([0, args.n])

# Equal physical scale in X and Y
ax.set_aspect("equal", adjustable="box")

plt.savefig(
    args.d + "/inputs/cdmats/patch_locations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



"""
plt.scatter(Locations[:,0],Locations[:,1])

plt.xlabel("X")
plt.ylabel("Y")

ax = plt.gca()
ax.set_xlim([0, scale_x])
ax.set_ylim([0, scale_y])
#plt.axis('scaled')
plt.savefig(args.d+"/inputs/cdmats/patch_locations.png",dpi=300)
plt.close()
"""
#=============================================================================#
# OUTPUT CDMATRIX.CSV
#=============================================================================#
np.savetxt(args.d+"/inputs/cdmats/cdmatrix.csv", ProbMatrix, delimiter=",")    

#=============================================================================#
# OUTPUT PatchVars.CSV
#=============================================================================#
#PatchVars headings
PVars_heading = ["PatchID","X","Y","SubpatchNO","K","K StDev","N0","Natal Grounds","Migration Out Grounds","Genes Initialize","Class Vars","Mortality Out","Mortality Out StDev","Mortality Back","Mortality Back StDev","Mortality Eggs","Mortality Eggs StDev","Migration Out Prob","Set Migration Out","Migration Back Prob","Straying Prob","Dispersal Prob","GrowthTemperatureOut","GrowthTemperatureOutStDev","GrowDaysOut","GrowDaysOutStDev","GrowthTemperatureBack","GrowthTemperatureBackStDev","GrowDaysBack","GrowDaysBackStDev","Capture Probability Out","Capture Probability Back","HabitatOut","HabitatBack","Fitness_AA","Fitness_Aa","Fitness_aa","Fitness_BB","Fitness_Bb","Fitness_bb","Fitness_AABB","Fitness_AaBB","Fitness_aaBB","Fitness_AABb","Fitness_AaBb","Fitness_aaBb","Fitness_AAbb","Fitness_Aabb","Fitness_aabb","comp_coef"]

default_values = [1,1,1,1,100,0,100,1,0,"random","classvars/ClassVars.csv",0,0,0,0,0,0,0,"N",0,0,1,0,0,0,0,0,0,0,0,"N","N",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


data = []
for i in range(n):
    data.append(copy.deepcopy(default_values))
    data[-1][0] = i+1


df = pd.DataFrame(columns = PVars_heading,data=data)

df["X"] = df["X"].astype(float)
df["Y"] = df["Y"].astype(float)
#PatchVars = pd.DataFrame(data=data,index=PVars_heading)

#Set the locations:
for i in range(n):
    df.loc[i,"X"] = Locations[i][0]
    df.loc[i,"Y"] = Locations[i][1]

df.to_csv(args.d+"/inputs/patchvars/PatchVars.csv",index=False)




