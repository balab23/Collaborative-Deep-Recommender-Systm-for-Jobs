import pickle
import pandas 
import numpy 

membership=pickle.load( open( "/home/itdept/Evaluation/All_Data/k_clusters.txt", "rb" ) )
#print "membership matrix"
membership=numpy.asarray(membership)
#print membership
data=pickle.load( open( "/home/itdept/Evaluation/All_Data/uservector.txt", "rb" ) )

#data=data.as_matrix()
data=data.astype(numpy.int)
#print data
users=data[:,0]
indices=list(data[:,0])


def select_users(user,threshold):
	li=[]
	i=0
	index=indices.index(user)
	cl=membership[index]	
	selected_users = numpy.where(membership==cl)[0]
	i=0
	selected_users=users[selected_users]
	print "selected are "
	print selected_users
	create_ratings(selected_users)
	fi=open('/home/itdept/Evaluation/All_Data/selected_users','wb')
	pickle.dump(selected_users,fi)
	#return ratings

def create_ratings(users):
	df=pandas.read_csv('/home/itdept/interactions.tsv', sep='\t',index_col=None)
#	print "interactions is"
#	print df
	matrix=df.as_matrix().astype(numpy.int)
	rating=[]
	i=0
	count=0
	ra=df.loc[df['user_id'].isin(users)]
	print ra
	pickle.dump(ra,open("/home/itdept/Evaluation/All_Data/ratings.txt","wb"))
	#return ratings
select_users(165,0.06)