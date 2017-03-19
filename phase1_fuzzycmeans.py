import skfuzzy
import pickle
import pandas as pd
import numpy as np

data=pickle.load( open( "userstest.txt", "rb" ) )
data=data.as_matrix()
data=data.astype(np.int)
#print data
data=np.transpose(data)
i=1
#while i <= 2:
	
cntr, u, u0, d, jm, p, fpc = skfuzzy.cmeans(data, 8, 2, 0.005, 1000, init=None, seed=None)
#print "u is :"
#print u
#print np.shape(data)
#print np.shape(u)
#print np.shape(cntr)
#print "fpc of i = "+str(i)+" is : "+str(fpc) 
#i+=1
fi=open("cluster_membership.txt","wb")
pickle.dump(u,fi)