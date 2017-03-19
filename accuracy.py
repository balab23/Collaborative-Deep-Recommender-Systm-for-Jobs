import pickle
import pandas as pd 
import numpy as np
import scipy.io as sio



test_user  = pickle.load(open('/home/itdept/Evaluation/All_Data/test_users.txt','rb'))
X = sio.loadmat('/home/itdept/Evaluation/All_Data/reccomendations.mat')
all_ratings = sio.loadmat('/home/itdept/Evaluation/ratings.mat')
all_ratings = pandas.DataFrame(all_ratings['R'])




X = X['binary_jobs']

job_ids = sio.loadmat('/home/itdept/Evaluation/All_Data/jobids.mat')
job_ids = job_ids['jid']
recj = X*job_ids
recj = pandas.DataFrame(recj)
f  = file('/home/itdept/Evaluation/All_Data/recommendation_jobs.txt','wb')
pickle.dump(recj,f)

recj = recj[recj[0].isin(test_user)]
t = recj.sort([0])
t = np.asarray(t)
Y = all_ratings[all_ratings[0].isin(test_user)]
Yt = Y.sort([0])
temp = np.asarray(Yt)

count =0
correct =0

for i in range(0,temp.shape[0]) :
	ind = (temp[i]>0)&(temp[i]!=1)
	count= count + len(t[i][ind])
	correct = correct +sum(t[i][ind]>0) + 1

acc = float(correct)/float(count)



temp = np.asarray(Yt.loc[36])











misprediction=len(R[R == 4])
total  = len(R) - len(R[R == 0]) 

accuracy = (total - misprediction)/total *100 











