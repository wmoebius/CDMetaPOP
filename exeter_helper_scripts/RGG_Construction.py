# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:49:56 2026

@author: thoma
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
import argparse
import os

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

args = parser.parse_args()
print(args.d)

if args.r != 0:
    np.random.seed(args.r)


#Directory where the inputs are going to be placed.
if args.d != "":
    if not os.path.isdir(str(args.d)):
        os.makedirs(str(args.d))
        os.makedirs(str(args.d)+"/classvars")
        os.makedirs(str(args.d)+"/cdmats")
        os.makedirs(str(args.d)+"/patchvars")
        os.makedirs(str(args.d)+"/popvars")

else:
    raise ValueError("Please specify a directory to save data using the -d flag.")

if args.i != "":
    if not os.path.isdir(str(args.i)):
        raise ValueError("The directory specified for the -i flag does not exist.")
    else:
        #Copy ClassVars, PopVars, and RunVars to the new directory.
        os.system("cp "+str(args.i)+"/classvars/ClassVars.csv "+str(args.d)+"/classvars/ClassVars.csv")
        os.system("cp "+str(args.i)+"/popvars/PopVars.csv "+str(args.d)+"/popvars/PopVars.csv")
        os.system("cp "+str(args.i)+"/RunVars.csv "+str(args.d)+"/RunVars.csv")


#=============================================================================#
# PARAMETERS
#=============================================================================#
#number of individual patches
n = 10

#scale of the bounding box of patches
scale_x = 10
scale_y = 15

#=============================================================================#
# RANDOMLY CONSTRUCT LOCATIONS
#=============================================================================#
Locations = []

for i in range(n):
    x = random.random()*scale_x
    y = random.random()*scale_y
    
    Locations.append((x,y))
    
Locations = np.asarray(Locations)
print("Locations:",Locations)


#=============================================================================#
# CONSTUCT COST MATRIX
#=============================================================================#
DistMatrix = np.zeros(shape=(n,n))

for i in range(len(DistMatrix)):
    for j in range(i,len(DistMatrix)):
        
        
        distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                           (Locations[i][1]-Locations[j][1])**2)
        
        DistMatrix[i][j] = DistMatrix[j][i] = distance
        
        


#=============================================================================#
# PLOTTING
#=============================================================================#
plt.scatter(Locations[:,0],Locations[:,1])

plt.xlabel("X")
plt.ylabel("Y")

ax = plt.gca()
ax.set_xlim([0, scale_x])
ax.set_ylim([0, scale_y])
#plt.axis('scaled')

plt.show()

#=============================================================================#
# OUTPUT CDMATRIX.CSV
#=============================================================================#
np.savetxt(args.d+"/cdmats/cdmatrix.csv", DistMatrix, delimiter=",")    

#=============================================================================#
# OUTPUT PatchVars.CSV
#=============================================================================#
#PatchVars headings
PVars_heading = ["PatchID","X","Y","SubpatchNO","K","K StDev","N0","Natal Grounds","Migration Out Grounds","Genes Initialize","Class Vars","Mortality Out","Mortality Out StDev","Mortality Back","Mortality Back StDev","Mortality Eggs","Mortality Eggs StDev","Migration Out Prob","Set Migration Out","Migration Back Prob","Straying Prob","Dispersal Prob","GrowthTemperatureOut","GrowthTemperatureOutStDev","GrowDaysOut","GrowDaysOutStDev","GrowthTemperatureBack","GrowthTemperatureBackStDev","GrowDaysBack","GrowDaysBackStDev","Capture Probability Out","Capture Probability Back","HabitatOut","HabitatBack","Fitness_AA","Fitness_Aa","Fitness_aa","Fitness_BB","Fitness_Bb","Fitness_bb","Fitness_AABB","Fitness_AaBB","Fitness_aaBB","Fitness_AABb","Fitness_AaBb","Fitness_aaBb","Fitness_AAbb","Fitness_Aabb","Fitness_aabb","comp_coef"]

default_values = [1,1,1,1,100,0,100,1,0,"random","classvars/ClassVars.csv",0,0,0,0,0,0,0,"N",0,0,1,0,0,0,0,0,0,0,0,"N","N",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


data = []
for i in range(n):
    data.append(copy.deepcopy(default_values))
    data[-1][0] = i


df = pd.DataFrame(columns = PVars_heading,data=data)

df["X"] = df["X"].astype(float)
df["Y"] = df["Y"].astype(float)
#PatchVars = pd.DataFrame(data=data,index=PVars_heading)

#Set the locations:
for i in range(n):
    df.loc[i,"X"] = Locations[i][0]
    df.loc[i,"Y"] = Locations[i][1]

df.to_csv(args.d+"/patchvars/PatchVars.csv",index=False)




