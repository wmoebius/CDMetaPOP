
#Wolfram's initial helper scripts
uv run generate_matrix_ai.py --f 0.5
sh run_scenario.sh Test1
uv run analysis_ai.py Test1


#Tom's scripts for tracking individuals across locations

1) A script which checks that individuals can be identified uniquely with the last three parts of their ID's, by performing checks such that gender doesn't flip and the age doesn't advance unusually.
uv run track_individuals_CheckIDUniqueness_ai.py -d $directory with the output csv files$

2) A script that tracks the location of all individuals in all generations. Currently produces a histogram at the end of lifetimes
uv run track_individuals_LocationAllGens_ai.py -d $directory with the output csv files$

#Tom's scripts for constructing and plotting pedigree trees.

1) A preliminary script which gathers all the pedigree information of individuals and produces an output file for later plotting.
uv run track_individuals_PedigreeTreePrelim_ai.py -d $directory with the csv files$

2) A plotting script which creates a number of pedigree trees, n.
uv run track_individuals_PedigreeTreePlotting_ai.py -d $directory with the csv files$
