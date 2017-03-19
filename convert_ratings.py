import pickle
import pandas 
import numpy 
import scipy.io as sio

ratings=pickle.load( open( "/home/itdept/Evaluation/All_Data/ratings.txt", "rb" ) )
print len(ratings)
jobs=pandas.unique(ratings.item_id.ravel())
print jobs
initial_jobs=pandas.read_csv("/home/itdept/items.tsv",delimiter="\t",index_col=None)
initial_jobs=initial_jobs.loc[initial_jobs['id'].isin(jobs)]
clustered_jobs=initial_jobs.loc[initial_jobs['active_during_test'] == 1]
jobs=pandas.unique(clustered_jobs.id.ravel())
jobs.sort()
pickle.dump(cluster_jobs,open('/home/itdept/Evaluation/All_Data/clustered_jobs.txt','wb'))
print jobs
users=pandas.unique(ratings.user_id.ravel())

#print len(pandas.unique(ratings.user_id.ravel()))
#print len(jobs)
#print len(users)

str_jobs=jobs.astype(str)
#print str_jobs
str_jobs=numpy.hstack(("user_id",str_jobs))
#print str_jobs
#final_ratings=pandas.DataFrame(columns=str_jobs)
#print final_ratings
#final_ratings=numpy.tile(0,(users.shape[0],jobs.shape[0]+1))
#final_ratings=numpy.memmap('ratingsmemap.txt',dtype='int32',mode='w+',shape=(users.shape[0],jobs.shape[0]+1))
#final_ratings[:,0]=users
#print "final_ratings is "
#print final_ratings
final_ratings=numpy.tile(0,(users.shape[0],jobs.shape[0]+1))
i=0
for user in users:
	print i
	u_jobs=ratings.loc[ratings['user_id']==user]
	u_jobs=u_jobs.loc[u_jobs['item_id'].isin(jobs)]['item_id']
	print "ujobs:"
	print list(u_jobs)
	u_ratings=ratings.loc[ratings['user_id']==user]['interaction_type']
	temp=numpy.tile(0,jobs.shape[0])
	print "ndx:"
	ndx = numpy.searchsorted(jobs, list(u_jobs))
	print ndx
	temp[ndx]=list(u_ratings)
	replace={1:2,2:3,3:4,4:1}
	mp = numpy.arange(0,4+1)
	mp[replace.keys()] = replace.values()
	temp = mp[temp]
	temp=numpy.hstack((user,temp))
	final_ratings[i,:]=temp
	print "done " + str(i)
	i+=1
	

	
	

#numpy.savetxt("full_ratings.csv", final_ratings, delimiter=",")
sio.savemat('/home/itdept/Evaluation/All_Data/ratings.mat', {'f':final_ratings})