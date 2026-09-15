import numpy as np

#Running parameters
batch_num = 25
batch_size = 40


repeats = int(batch_num * batch_size)


#Landscape Parameters
#node number
n = 20

#Probability distribution for edge weights
ProbDist = "Power" #Exponential, Power

#Parameters for probability distribution
param1 = 4


SaveDirName = ("SaveFiles/" +
"efficientbatches_n%d_ProbDist_%s_param1_%s_repeats_%d" % (n, str(ProbDist), str(param1), repeats))