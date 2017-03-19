import skfuzzy
import pickle
import pandas 
import numpy 

membership=pickle.load( open( "cluster_membership.txt", "rb" ) )
print "membership matrix"
print sum(membership[:,1])