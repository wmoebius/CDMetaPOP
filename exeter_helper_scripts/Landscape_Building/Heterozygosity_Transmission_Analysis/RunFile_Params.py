import numpy as np

#Running parameters
repeats = 20


#Landscape Parameters
#node number
n = 20

#Probability distribution for edge weights
ProbDist = "Power" #Exponential, Power

#Parameters for probability distribution
param1 = 2


SaveDirName = ("SaveFiles/" +
"n%d_ProbDist_%s_param1_%s_repeats_%d" % (n, str(ProbDist), str(param1), repeats))