import pandas 
import numpy 
import scipy.io as sio

all_jobs  = pandas.read_csv("modified_items.csv")
train = scipy.io.loadmat("train.mat")
train =train['train']
jobs = all_jobs[all_jobs['id'].isin(train[:,1])]
jobs=jobs.fillna(0)
li=pandas.unique(jobs.country.ravel())
li=list(li)
li=li[0:4]
jobs=jobs.replace(li,[1,2,3,4])
print jobs
job_matrix = jobs.as_matrix()
sio.savemat('jobs.mat', {'X':job_matrix})