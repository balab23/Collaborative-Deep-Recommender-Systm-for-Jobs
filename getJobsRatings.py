import pickle
import pandas 
import numpy 
import scipy.io as sio


df=pandas.read_csv('/home/itdept/interactions.tsv',delimiter='\t',index_col=None)
udf=pandas.read_csv('/home/itdept/users.tsv',delimiter='\t',index_col=None)
users=pandas.unique(df.user_id.ravel())
jobs=pandas.unique(df.item_id.ravel())
total_users=users[0:100]
train_users=users[0:70]
test_users=users[71:100]
ratings_full=df.loc[df['user_id'].isin(total_users)]
ratings_train=df.loc[df['user_id'].isin(train_users)]
pickle.dump(total_users,open('/home/itdept/Evaluation/All_Data/total_users.txt','wb'))
pickle.dump(train_users,open('/home/itdept/Evaluation/All_Data/train_users.txt','wb'))
pickle.dump(test_users,open('/home/itdept/Evaluation/All_Data/test_users.txt','wb'))
pickle.dump(ratings_full,open('/home/itdept/Evaluation/All_Data/ratings_full.txt','wb'))
pickle.dump(ratings_train,open('/home/itdept/Evaluation/All_Data/ratings_train.txt','wb'))
final_users=udf.loc[udf['id'].isin(total_users)]
pickle.dump(final_users,open('/home/itdept/Evaluation/All_Data/final_users.txt','wb'))






