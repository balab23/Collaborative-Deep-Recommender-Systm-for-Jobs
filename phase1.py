import pandas 
import pickle

df=pandas.DataFrame.from_csv('C:\Minor Project\Dataset\users.tsv', sep='\t',index_col=None)
print df

li=pandas.unique(df.country.ravel())
li=list(li)
li=li[0:4]
#for l in li:
df=df.replace(li,[1,2,3,4])
print df
filee = open("userstest.txt", 'wb')
pickle.dump(df,filee)