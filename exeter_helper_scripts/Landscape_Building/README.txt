Here is an example layout for running this:

uv run Landscape_Construction.py -i param_variation_environment/inputs/ -n 20 -ProbDist Exponential -param1 50

uv run ../../src/CDmetaPOP.py RGG_n20_ProbDist_Exponential_param1_50.0_nonperiodic_seed0/inputs RunVars.csv ../outputs/raw/

uv run Phylogeography_ai.py -d RGG_n20_ProbDist_Exponential_param1_50.0_nonperiodic_seed0/outputs/raw/1787060779/run0batch0mc0species0/ -g L0A0A0 -PID 1 --counts --unique


More detail:

Landscape_Construction
Landscape_Construction creates the cdmat and populates the rest of the inputs by taking tagets in the -i directory. If you want to give the output directory a particular name, employ -d to give that name. Else, a name will be generated based on the other arguments.

Arguments:
-d name of output directory (default is to name based on the other params)

-i the name of the input directory

-n the number of nodes for the RGG and 1D cases, or the edge length for the square and hexagonal lattices

-type The type of network: RGG, SLattice (square lattice), HLattice (Hexagonal lattice), 1D

-ProbDist Only relevant for the RGG: if a number is given it's the RGG radius, otherwise Exponential, Gaussian and Linear can be used for weights.

-param1 So far just acts as the exponential parameter for the Exponential probdist (e^-param1*distance). One day will also be used for Gaussian. Defaults to 1

-r the randomseed (an integer) for the sake of making an RGG. Defaults to 0.

-periodic Describes the periodicity of the system. Defauls to 'False', but can also be x, y or xy. Currently only x has periodicity, and only words for the linear, HLattice and SLattice.




Phylogeography_ai.py

Creates a png of the phylogeographical plot I care about, wherein the number of individuals in the genetic history of a target genotype in a target location is traced back.

Arguments:
-d the directory of the specific mcrun with the ind*.csv file

-g The target genotype. L0A0A0, L0A0A1, L0A1A1, L1... etc

-PID The patch ID of the patch we focus on.