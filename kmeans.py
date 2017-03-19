import skfuzzy
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_samples, silhouette_score

data=pickle.load( open( "/home/itdept/Evaluation/All_Data/uservector.txt", "rb" ) )
#data =data.fillna(0)
data=data[1:,:]
print data
#data=data.as_matrix()
#print data
#data=data.astype(np.int)
#data=data.astype(np.double)
#kmeans=KMeans()p
#kmeans.fit(data)

k_range=range(1,8)
results=KMeans(n_clusters=2).fit_predict(data)
print results
pickle.dump(results,open("/home/itdept/Evaluation/All_Data/k_clusters.txt","wb"))

from sklearn import cluster
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(data)

from matplotlib import pyplot
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
for i in range(2):
    # select only data observations with cluster label == i
    ds = data[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()

'''
range_n_clusters = [2, 3, 4]

for n_clusters in range_n_clusters:
	print n_clusters
	clusterer = KMeans(n_clusters=n_clusters, random_state=10)
	cluster_labels = clusterer.fit_predict(data)
	silhouette_avg = silhouette_score(data, cluster_labels)
	print "For n_clusters ="+str(n_clusters)+"The average silhouette_score is :"+str(silhouette_avg)
'''