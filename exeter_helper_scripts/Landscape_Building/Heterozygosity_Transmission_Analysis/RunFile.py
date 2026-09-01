import numpy as np
from RunFile_Params import n, ProbDist, param1,SaveDirName,repeats
import Landscape_Construction

import time

import subprocess
import os,shutil
import argparse


starttime = time.time()

#=============================================================================#
# ARGPARSER
#=============================================================================#
parser = argparse.ArgumentParser(
    prog = "RGG_Construction.py",
    description="Create a landscape for CDmetaPOP")



parser.add_argument("-i",
                    type=str,
                    default='default_inputs/inputs',
                    help='inputs directory for simulation')


args = parser.parse_args()





#=============================================================================#
# Saving Details
#=============================================================================#
if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Directory")

try:
    shutil.copy("RunFile_Params.py",SaveDirName)
except:
    print("RunFile_Params.py not found, not copied to SaveDirName")



print("Starting to create landscapes")
#Create the landscapes
Repeat_Directories = []
for rep in range(repeats):
    print(f"Creating landscape for repeat {rep}")
    directoryname = SaveDirName+"/Repeat_%d"%rep   
    Repeat_Directories.append(directoryname)
    Landscape_Construction.main(n=n, ProbDist=ProbDist, param1=param1, d=directoryname, i=args.i,r=rep)
print("Finished creating landscapes")



print("Starting landscape simulations")
#Execute the landscape simulations in this format:
# uv run ../../src/CDmetaPOP.py RGG_n20_ProbDist_Power_param1_4.0_nonperiodic_seed0/inputs RunVars.csv ../outputs/raw/
plist = []

for i in Repeat_Directories:
    p=subprocess.Popen(['nice','-n','19','uv','run','../../../src/CDmetaPOP.py',str(i)+'/inputs', 'RunVars.csv', '../outputs/raw/'])
    plist.append(p)

for p in plist:
    p.wait()


print("Finished landscape simulations")


endtime = time.time()
print(f"Total time taken: {endtime - starttime} seconds")
