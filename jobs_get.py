import pickle
import pandas 
import numpy as np
import scipy.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


f = file("/home/itdept/Evaluation/All_Data/clustered_jobs.txt","r+")
jobs = pickle.load(f)
job  = jobs.fillna('0')
ddff=pickle.load(open('/home/itdept/Evaluation/All_Data/ratings.txt','rb'))
print ddff
train_users=pickle.load(open('/home/itdept/Evaluation/All_Data/train_users.txt','rb'))
train_jobs=ddff.loc[ddff['user_id'].isin(train_users)]['item_id'].as_matrix()
train_jobs.sort()
test_users = pickle.load(open('/home/itdept/Evaluation/All_Data/test_users.txt','rb'))
#job=job.loc[job['id'].isin(train_jobs)]
job = job.sort(columns='id')

job_ids = np.asarray(job['id'])
job_ids = np.append(np.tile(1,(1,1)),job_ids)
sio.savemat('/home/itdept/Evaluation/All_Data/jobids.mat',{'jid':job_ids})

ratings = sio.loadmat('/home/itdept/Evaluation/ratings.mat')
rating = pandas.DataFrame(ratings['R'])
rating.ix[rating[0].isin(test_users),range(1,rating.shape[1])] = 0
ratings = np.asarray(rating)
sio.savemat('/home/itdept/Evaluation/All_Data/ratings_after.mat', {'R':ratings})

title = job['title'].replace(',',' ',regex = True)
title=title.tolist()
tfidf = TfidfVectorizer(max_df =0.95,min_df = 2 )
vecs = tfidf.fit_transform(title)
title_vector = np.array(vecs.todense())


discipline = job['discipline_id']
discipline =discipline.as_matrix()
discipline = discipline.astype(str)
vecs = tfidf.fit_transform(discipline)
discipline_vector = np.array(vecs.todense())


region = job['region']
region =region.as_matrix()
region = region.astype(str)
vecs = tfidf.fit_transform(region)
region_vector = np.array(vecs.todense())

tfidf= TfidfVectorizer(max_df=0.95, min_df=2)

industry=job['industry_id'].tolist()
#print industry
#industry=industry.astype(str)
industry = list(map(str,industry ))
industry_vector=tfidf.fit_transform(industry)
#print industry_vector.shape
industry_vector=np.array(industry_vector.todense())
print industry_vector


country=job['country'].tolist()
#print country
#industry=industry.astype(str)
country = list(map(str,country ))
country_vector=tfidf.fit_transform(country)
#print country_vector.shape
country_vector=np.array(country_vector.todense())
print country_vector

tags = job['tags'].replace(',',' ',regex = True)
tags=tags.tolist()
tfidf = TfidfVectorizer(max_df =0.95,min_df = 2 )
vecs = tfidf.fit_transform(tags)
tags_vector = np.array(vecs.todense())


pca = PCA(n_components=10)
tags_v = pca.fit_transform(tags_vector)
title_v = pca.fit_transform(title_vector)


final_features = np.append(title_v,region_vector,axis=1)
final_features = np.append(final_features,country_vector,axis =1)
final_features = np.append(final_features,industry_vector,axis =1)
final_features = np.append(final_features,tags_v,axis =1)


other_list = ['id','career_level','employment']
other_features = np.asarray(job[other_list])
user_features = np.append(other_features,final_features,axis =1)
sio.savemat('/home/itdept/Evaluation/All_Data/jobs.mat',{'X':user_features})







