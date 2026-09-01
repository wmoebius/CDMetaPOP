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



def Find_Connected_Components(Locations, radius):
    """
    Find connected components of points.

    Two points are directly connected if their Euclidean
    distance is <= radius.

    Returns
    -------
    Components : list of lists
        Each element is a list containing the points belonging
        to one connected component.
    """

    n = len(Locations)

    # Calculate pairwise distances
    Diff = Locations[:, np.newaxis, :] - Locations[np.newaxis, :, :]
    Distances = np.sqrt(np.sum(Diff**2, axis=2))

    # Adjacency matrix
    Connected = Distances <= radius

    # Keep track of which points have been visited
    Visited = np.zeros(n, dtype=bool)

    Components = []

    for i in range(n):

        if Visited[i]:
            continue

        # Start a new component
        Component_Indices = []
        Stack = [i]
        Visited[i] = True

        while Stack:

            current = Stack.pop()
            Component_Indices.append(current)

            # Find all unvisited neighbours
            Neighbours = np.where(Connected[current] & ~Visited)[0]

            for neighbour in Neighbours:
                Visited[neighbour] = True
                Stack.append(neighbour)

        # Convert indices to actual points
        Components.append(Locations[Component_Indices].tolist())

    return Components




















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
                    help='New random seed: defaults to 0.')

parser.add_argument("-type",
                    type=str,
                    default='RGG',
                    help='Type of landscape to create: RGG, SLattice (Square Lattice), HLattace (Hexagonal Lattics) or 1D',
                    choices=['RGG','SLattice',"HLattice","1D"])

parser.add_argument("-periodic",
                    type = str,
                    default = "False",
                    choices = ["False","x","y","xy"],
                    help='Whether to use periodic boundary conditions: False, x (x-periodic only), y (y-periodic only), xy (both x and y periodic)')

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



parser.add_argument("-sticky_radius",
                    type=float,
                    default = 0,
                    help='Radius about where we combine nodes into larger nodes.')


def main(d='', i='', r=0, graph_type='RGG', periodic='False', n=10, ProbDist=0.1, param1=1, param2=1, sticky_radius=0):
    ##########################################################
    # PROCESS ARGS
    ##########################################################

    np.random.seed(r)


    #Directory where the inputs are going to be placed.
    outdir = d

    if outdir == "":
        #Create a directory name based on the args
        
            if graph_type == "RGG":
                outdir = "RGG_n%d_ProbDist_%s" % (n, str(ProbDist))

                if ProbDist == "Exponential":
                    outdir += "_param1_%s" % (str(param1))

                if ProbDist == 'Power':
                    outdir += "_param1_%s" % (str(param1))
        
            if graph_type == "SLattice":
                outdir = "SLattice_n%d" % (n)
        
            if graph_type == "HLattice":
                outdir = "HLattice_n%d" % (n)
        
            if graph_type == "1D":
                outdir = "1D_n%d" % (n)
        


            if periodic != "False":
                outdir += "_periodic_%s" % (periodic)

            else:
                outdir += "_nonperiodic"


            if graph_type == "RGG":
                if sticky_radius > 0:
                    outdir += "_sticky_radius_%s" % (str(sticky_radius))
                outdir += "_seed%d" % (r)


    if not os.path.isdir(str(outdir)):
        os.makedirs(str(outdir))
        os.makedirs(str(outdir)+"/inputs/classvars")
        os.makedirs(str(outdir)+"/inputs/cdmats")
        os.makedirs(str(outdir)+"/inputs/patchvars")
        os.makedirs(str(outdir)+"/inputs/popvars")

        os.makedirs(str(outdir)+"/outputs/raw")
        os.makedirs(str(outdir)+"/outputs/analysed")

    #Directory where the ClassVars, PopVars, and RunVars are located.
    if i != "":
        if not os.path.isdir(str(i)):
            raise ValueError("The directory specified for the -i flag does not exist.")
        else:
            #Copy ClassVars, PopVars, and RunVars to the new directory.
            os.system("cp "+str(i)+"/classvars/ClassVars.csv "+str(outdir)+"/inputs/classvars/ClassVars.csv")
            os.system("cp "+str(i)+"/popvars/PopVars.csv "+str(outdir)+"/inputs/popvars/PopVars.csv")
            os.system("cp "+str(i)+"/RunVars.csv "+str(outdir)+"/inputs/RunVars.csv")


    #=============================================================================#
    # PARAMETERS
    #=============================================================================#
    #number of individual patches
    n = n




    #=============================================================================#
    # CONSTUCT DISTANCE MATRIX GIVEN THE graph_type OF LANDSCAPE
    #=============================================================================#
    if graph_type == "RGG":

        Locations = []

        for i in range(n):
            x = np.random.random()#*scale_x
            y = np.random.random()#*scale_y
            
            Locations.append((x,y))
            
        Locations = np.asarray(Locations)
        print("Locations:",Locations)

        #If sticky_radius is greater than 0, find connected components and combine them into single nodes.
        if sticky_radius > 0:
            Components = Find_Connected_Components(Locations, sticky_radius)


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




        DistMatrix = np.zeros(shape=(len(Locations),len(Locations)))

        for i in range(len(DistMatrix)):
            for j in range(i,len(DistMatrix)):
                
                
                distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                                (Locations[i][1]-Locations[j][1])**2)
                
                DistMatrix[i][j] = DistMatrix[j][i] = distance
            
            
    if graph_type == "SLattice":
        #Create a square lattice of patches
        side_length = n
        x_coords = np.arange(side_length)
        y_coords = np.arange(side_length)
        
        Locations = np.array([(x, y) for x in x_coords for y in y_coords])
        print("Locations:",Locations)
        DistMatrix = np.zeros(shape=(len(Locations),len(Locations)))
        
        for i in range(len(DistMatrix)):
            for j in range(i,len(DistMatrix)):
                if periodic == "False":
                    distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                                    (Locations[i][1]-Locations[j][1])**2)
                
                elif periodic == "x":

                    max_x = max(x for x, y in Locations)

                    x_dist = min(abs(Locations[i][0]-Locations[j][0]), (max_x+1) - abs(Locations[i][0]-Locations[j][0]))
                    y_dist = abs(Locations[i][1]-Locations[j][1])
                    distance = np.sqrt(x_dist**2 + y_dist**2)
                
                DistMatrix[i][j] = DistMatrix[j][i] = distance

    if graph_type == "HLattice":
        #Create a hexagonal lattice of patches
        side_length = n
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

                if periodic == "False":
                    distance = np.sqrt((Locations[i][0]-Locations[j][0])**2 + 
                                    (Locations[i][1]-Locations[j][1])**2)
                
                elif periodic == "x":

                    max_x = max(x for x, y in Locations)

                    x_dist = min(abs(Locations[i][0]-Locations[j][0]), max_x - abs(Locations[i][0]-Locations[j][0]))
                    y_dist = abs(Locations[i][1]-Locations[j][1])
                    distance = np.sqrt(x_dist**2 + y_dist**2)

                    

                DistMatrix[i][j] = DistMatrix[j][i] = distance

    if graph_type == "1D":
        #Create a 1D organisation of patches on the unit circle
        Locations = []
        for i in range(n):

            if periodic != "False":
                angle = 2 * np.pi * i / n
            else:
                angle = 2 * np.pi * i / (n+1)
            x = np.cos(angle)
            y = np.sin(angle)
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


    if graph_type == "RGG":
        if type(ProbDist) == float:
            radius = ProbDist
            ProbMatrix = (DistMatrix <= radius).astype(float)

            

        elif ProbDist == "Linear":
            ProbMatrix = 1 - (DistMatrix / DistMatrix.max())

        elif ProbDist == "Exponential":
            ProbMatrix = np.exp(-param1 * DistMatrix)

        elif ProbDist == "Gaussian":
            sigma = DistMatrix.mean()
            ProbMatrix = np.exp(-DistMatrix**2 / (2 * sigma**2))

        elif ProbDist == "Power":
            ProbMatrix = 1 / (DistMatrix**param1)
            ProbMatrix[DistMatrix == 0] = 0  # Avoid division by zero


    if graph_type == "SLattice":
        ProbMatrix = (DistMatrix <= 1).astype(float)

    if graph_type == "HLattice":
        ProbMatrix = (DistMatrix <= 1.1).astype(float)

    if graph_type == "1D":
        ProbMatrix = (DistMatrix <= 1.1 * 2 * np.sin(np.pi / n)).astype(float)


    #Normalise
    np.fill_diagonal(ProbMatrix, 0)
    # Row sums
    row_sums = ProbMatrix.sum(axis=1, keepdims=True)


    # Only normalise rows with at least one possible destination
    non_empty = row_sums[:, 0] > 0

    ProbMatrix[non_empty] /= row_sums[non_empty]

    #If all zero for a row, make that connect to itself.
    if type(ProbDist) == float:
        zero_rows = np.sum(ProbMatrix, axis=1) == 0
        ProbMatrix[zero_rows, np.arange(ProbMatrix.shape[0])[zero_rows]] = 1


    """
    ###############
    # Eigenvalues
    ###############
    if args.type == "RGG":
        Eigenvalues = np.linalg.eigvals(ProbMatrix)
        print("Eigenvalues:", Eigenvalues)
        print("Effective number of mixing steps:", 1 / (1 - max(abs(Eigenvalues[1:]))))  # Exclude the first eigenvalue which is always 1

    ###############
    """



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
    if sticky_radius == 0:
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=100,
            ax=ax,
            hide_ticks=False
        )
    else:
        """If nodes have been collated, make them larger"""
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=100 * Locations_Sizes**2,  # Scale node size by the number of combined nodes
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

    if graph_type == "RGG":
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        for i, txt in enumerate(np.arange(len(Locations))+1):
            ax.annotate(txt, (Locations[i][0], Locations[i][1]), fontsize=8, ha='center', va='center',color='red')

    if graph_type == "SLattice":
        ax.set_xlim([-0.5, n-0.5])
        ax.set_ylim([-0.5, n-0.5])

    if graph_type == "HLattice":
        ax.set_xlim([-0.5, n - 0.5])
        ax.set_ylim([-0.5, n * (np.sqrt(3)/2) - 0.5])

    if graph_type == '1D':
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

    # Equal physical scale in X and Y
    ax.set_aspect("equal", adjustable="box")

    plt.savefig(
        outdir + "/inputs/cdmats/patch_locations.png",
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
    plt.savefig(outdir+"/inputs/cdmats/patch_locations.png",dpi=300)
    plt.close()
    """
    #=============================================================================#
    # OUTPUT CDMATRIX.CSV
    #=============================================================================#
    np.savetxt(outdir+"/inputs/cdmats/cdmatrix.csv", ProbMatrix, delimiter=",")    

    #=============================================================================#
    # OUTPUT PatchVars.CSV
    #=============================================================================#
    #PatchVars headings
    PVars_heading = ["PatchID","X","Y","SubpatchNO","K","K StDev","N0","Natal Grounds","Migration Out Grounds","Genes Initialize","Class Vars","Mortality Out","Mortality Out StDev","Mortality Back","Mortality Back StDev","Mortality Eggs","Mortality Eggs StDev","Migration Out Prob","Set Migration Out","Migration Back Prob","Straying Prob","Dispersal Prob","GrowthTemperatureOut","GrowthTemperatureOutStDev","GrowDaysOut","GrowDaysOutStDev","GrowthTemperatureBack","GrowthTemperatureBackStDev","GrowDaysBack","GrowDaysBackStDev","Capture Probability Out","Capture Probability Back","HabitatOut","HabitatBack","Fitness_AA","Fitness_Aa","Fitness_aa","Fitness_BB","Fitness_Bb","Fitness_bb","Fitness_AABB","Fitness_AaBB","Fitness_aaBB","Fitness_AABb","Fitness_AaBb","Fitness_aaBb","Fitness_AAbb","Fitness_Aabb","Fitness_aabb","comp_coef"]

    default_values = [1,1,1,1,100,0,100,1,0,"random","classvars/ClassVars.csv",0,0,0,0,0,0,0,"N",0,0,1,0,0,0,0,0,0,0,0,"N","N",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


    default_K = default_values[4]
    default_N0 = default_values[6]

    data = []
    for i in range(len(Locations)):
        data.append(copy.deepcopy(default_values))
        data[-1][0] = i+1

        if sticky_radius > 0:
            data[-1][4] = default_K * Locations_Sizes[i]  # Scale K by the number of combined nodes
            data[-1][6] = default_N0 * Locations_Sizes[i]  # Scale N0 by the number of combined nodes

    df = pd.DataFrame(columns = PVars_heading,data=data)

    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    #PatchVars = pd.DataFrame(data=data,index=PVars_heading)

    #Set the locations:
    for i in range(len(Locations)):
        df.loc[i,"X"] = Locations[i][0]
        df.loc[i,"Y"] = Locations[i][1]

    df.to_csv(outdir+"/inputs/patchvars/PatchVars.csv",index=False)






if __name__ == "__main__":
    args = parser.parse_args()

    main(
        d=args.d,
        i=args.i,
        r=args.r,
        graph_type=args.type,
        periodic=args.periodic,
        n=args.n,
        ProbDist=args.ProbDist,
        param1=args.param1,
        param2=args.param2,
        sticky_radius=args.sticky_radius
    )

