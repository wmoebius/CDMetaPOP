import numpy as np

#Running parameters
batch_num = 3
batch_size = 2


repeats = int(batch_num * batch_size)


#Landscape Parameters
#node number
n = 20

#Probability distribution for edge weights
ProbDist = "Power" #Exponential, Power

#Parameters for probability distribution
param1 = 2


SaveDirName = ("SaveFiles/" +
"n%d_ProbDist_%s_param1_%s_repeats_%d" % (n, str(ProbDist), str(param1), repeats))