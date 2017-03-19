import pickle
import pandas 
import numpy as np
import scipy.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

udf  = pickle.load(open('/home/itdept/Evaluation/All_Data/final_users.txt','rb'))
udf = udf.fillna(0)


jobroles = udf['jobroles'].replace(',',' ',regex = True)
jobroles=jobroles.tolist()
tfidf = TfidfVectorizer(max_df =0.95,min_df = 2 )
vecs = tfidf.fit_transform(jobroles)
jobroles_vector = np.array(vecs.todense())


discipline = udf['discipline_id']
discipline =discipline.as_matrix()
discipline = discipline.astype(str)
vecs = tfidf.fit_transform(discipline)
discipline_vector = np.array(vecs.todense())


region = udf['region']
region =region.as_matrix()
region = region.astype(str)
vecs = tfidf.fit_transform(region)
region_vector = np.array(vecs.todense())

tfidf= TfidfVectorizer(max_df=0.95, min_df=2)

industry=udf['industry_id'].tolist()
#print industry
#industry=industry.astype(str)
industry = list(map(str,industry ))
industry_vector=tfidf.fit_transform(industry)
#print industry_vector.shape
industry_vector=np.array(industry_vector.todense())
print industry_vector

country=udf['country'].tolist()
#print country
#industry=industry.astype(str)
country = list(map(str,country ))
country_vector=tfidf.fit_transform(country)
#print country_vector.shape
country_vector=np.array(country_vector.todense())
print country_vector

'''
final_features = np.append(jobroles_vector,discipline_vector,axis=1)
final_features = np.append(final_features,region_vector,axis =1)
final_features = np.append(final_features,country_vector,axis =1)
final_features = np.append(final_features,industry_vector,axis =1)
'''

pca = PCA(n_components=10)
jobroles_v = pca.fit_transform(jobroles_vector)
#discipline_v = pca.fit_transform(discipline_vector)
#region_v = pca.fit_transform(region_vector)
#country_v =pca.fit_transform(country_vector)
#industry_v = pca.fit_transform(industry_vector)


final_features = np.append(jobroles_v,discipline_vector,axis=1)
final_features = np.append(final_features,region_vector,axis =1)
final_features = np.append(final_features,country_vector,axis =1)
final_features = np.append(final_features,industry_vector,axis =1)

other_list = ['id','career_level','experience_n_entries_class','experience_years_experience','experience_years_in_current','edu_degree']
other_features = np.asarray(udf[other_list])
user_features = np.append(other_features,final_features,axis =1)
f  = file("/home/itdept/Evaluation/All_Data/uservector.txt","wb")
pickle.dump(user_features,f)
f.close()

import matplotlib.pyplot as plt
pca = PCA(n_components=2)
plot_data= pca.fit_transform(user_features)
x = plot_data[:,0]
y =plot_data[:,1]
plt.scatter(x,y)
plt.show()


'''
pca = PCA(n_components='mle')
pca_features = pca.fit_transform(final_features)
other_list = ['id','career_level','experience_n_entries_class','experience_years_experience','experience_years_in_current','edu_degree']
other_features = np.asarray(udf[other_list])
user_features = np.append(other_features,pca_features,axis =1)
 f  = file("uservector2.txt","wb")
pickle.dump(user_features,f)
f.close()
'''






