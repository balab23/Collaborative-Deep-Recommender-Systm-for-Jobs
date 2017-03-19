from numpy import genfromtxt
import pandas
import skfuzzy as fuzz
import numpy as np
import pandas as pd
import pickle as pk

from pandas import DataFrame
df = DataFrame.from_csv("C:\Minor Project\Dataset\users.tsv", sep="\t")
#print df
data=df.as_matrix()
#data=dataset.astype(np.float)

#print len(data)
print data 

li=pd.unique(df.country.ravel())
li=li[0:4]
li=list(li)
print li
for word in li:
  #a=df[df['country']==word].index
  #for num in a:
  #	df.iloc[num,5]=li.index(word)
  df.replace({word:li.index(word)},regex=False)
f=file("finaluser","w+")
pk.dump(df,f)

print df['country'].head()


#cntr, u, u0, d, jm, p, fpc=fuzz.cluster.cmeans(data, 4, 2, error=0.005, maxiter=1000, init=None, seed=None)
#print np.shape(data)
