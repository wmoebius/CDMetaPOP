import numpy as np

#Running parameters
batch_num = 16#16
batch_size = 64


repeats = int(batch_num * batch_size)


#Landscape Parameters
#node number
n = 20

#Probability distribution for edge weights
ProbDist = "Power" #Exponential, Power

#Parameters for probability distribution
param1 = 1


SaveDirName = ("SaveFiles/" +
"Startgenes100_CORRECTEDTM_alphabetagamma_Loci_20_Emily_efficientbatches_n%d_ProbDist_%s_param1_%s_repeats_%d" % (n, str(ProbDist), str(param1), repeats))
