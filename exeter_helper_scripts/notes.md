
#Wolfram's initial helper scripts
uv run generate_matrix_ai.py --f 0.5
sh run_scenario.sh Test1
uv run analysis_ai.py Test1


#Tom's scripts for tracking individuals

1) A script which checks that individuals can be identified uniquely with the last three parts of their ID's, by performing checks such that gender doesn't flip and the age doesn't advance unusually.
uv run track_individuals_check_ai.py -d $directory with the output csv files$

2) A script that tracks the location of all individuals in the first generation
uv run track_individuals_gen1_ai.py -d $directory with the output csv files$

3) A script that tracks the location of all individuals in all generations. Currently produces a histogram at the end of lifetimes
uv run track_individuals_allgens_ai.py -d $directory with the output csv files$