import pickle
import pandas 
import numpy 
import scipy.io as sio
import pickle

udf  = pandas.read_csv('/home/itdept/users.csv',delimiter='\t')
udf = udf.as_matrix()
udf = udf[0:10000,0]
train_users =udf[0:7000]
test_users = udf[7001:9999]
df=pandas.read_csv('/home/itdept/interactions.tsv',delimiter='\t')
df=df.as_matrix()
train_interactions = df[df['user_id'].isin(train_users)]
test_interactions = df[df['user_id'].isin(test_users)]
train_interactions = train_interactions.as_matrix()
test_interactions = test_interactions.as_matrix()
sio.savemat('train.mat', {'train':train_interactions})
sio.savemat('test.mat', {'test':test_interactions})



