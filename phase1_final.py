import pickle
import pandas 
import numpy 

membership=pickle.load( open( "k_clusters.txt", "rb" ) )
#print "membership matrix"
membership=numpy.asarray(membership)
#print membership
data=pickle.load( open( "userstest.txt", "rb" ) )
data=data.as_matrix()
data=data.astype(numpy.int)
#print data
indices=list(data[:,0])

def select_users(user,threshold):
	li=[]
	#selected_users=[]
	i=0
#	print indices[0]
	index=indices.index(user)
#	print "index is"+str(index)
	cl=membership[index]
#	print "cluster "+str(cl)
		
	selected_users = numpy.where(membership==cl)[0]
#	print "selected is"
#	print selected_users
#	print len(selected_users)

	#print "size of selected users" +str(len(selected_users))
	create_ratings(selected_users)
	fi=open('selected_users','wb')
	pickle.dump(selected_users,fi)

def create_ratings(users):
	df=pandas.read_csv('C:\Minor Project\interactions.tsv', sep='\t',index_col=None)
#	print "interactions is"
#	print df
	matrix=df.as_matrix().astype(numpy.int)
	
	rating=[]
	i=0
	'''
	while i<matrix.shape[0]:
		if matrix[i][0] in users:
			rating.append(matrix[i])
		print i 
		i+=1 
	print "final ratings"
	print rating
	print rating.shape()
	'''
	count=0
	for user in users:
		count+=1
		if count==50:
			break
		print "user is "+str(indices[user])
		if indices[user] in matrix[:,0]:
			print "yes!!"
			rating.append(matrix[list(matrix[:,0]).index(indices[user])])

		#rating=list(set().union(rating,df.loc[df['user_id'] == user]))
		
	#print "rating is"
	#print rating
	pickle.dump(rating,open("ratings.txt","wb"))
select_users(9,0.06)