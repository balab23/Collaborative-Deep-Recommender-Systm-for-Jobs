import pickle
import pandas 
import numpy as np
import scipy.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer

udf  = pandas.read_csv('/home/itdept/users.tsv',delimiter='\t')
udf = udf.loc[0:6999]
udf = udf.fillna(0)

jobroles = udf['jobroles'].replace(',',' ',regex = True)
jobroles=jobroles.tolist()
print type(jobroles[0])
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
education=udf['edu_fieldofstudies'].astype(str).replace(',',' ',regex = True)
print education
tfidf= TfidfVectorizer(min_df=1)
education = education.tolist()
print type(education[0])
print education
education_vector=tfidf.fit_transform(education)
#education_vector=np.array(education_vector.todense())
#print education_vector
'''





