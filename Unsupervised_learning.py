from sklearn.cluster import KMeans,DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
data=pd.read_csv('train.csv')
df=pd.DataFrame(data)
#K-means Clustering
df1=df[['Item_Weight','Item_MRP']].copy()
new_df1=df1.copy()
obj=KMeans(n_clusters=6,random_state=0)
obj.fit(df1)
print(obj.predict(df1))
print(obj.cluster_centers_)
df1['Cluster'] = obj.labels_
print(df1)
plt.scatter(df1['Item_Weight'], df1['Item_MRP'], c=df1['Cluster'])
plt.xlabel("Item_Weight")
plt.ylabel("Item_MRP")
plt.title("Clusters Visualization")
plt.show()
sil_score = silhouette_score(new_df1, obj.labels_)
db_score = davies_bouldin_score(new_df1, obj.labels_)
ch_score = calinski_harabasz_score(new_df1, obj.labels_)
print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)
print("Calinski-Harabasz Index:", ch_score)
#Hierarchical Clustering
df2=df1[['Item_Weight','Item_MRP']].copy()
Z = linkage(df2, method='ward')
labels = fcluster(Z, t=6, criterion='maxclust')
df2['Cluster2'] = labels
print(df2)
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
sil_score = silhouette_score(new_df1, labels)
db_score = davies_bouldin_score(new_df1, labels)
ch_score = calinski_harabasz_score(new_df1, labels)
print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)
print("Calinski-Harabasz Index:", ch_score)
#DBSCAN
df3=df2[['Item_Weight','Item_MRP']].copy()
db = DBSCAN(eps=2.5, min_samples=2)
db.fit_predict(df3)
df3['Cluster3']=db.labels_
print(df3)
plt.scatter(df3['Item_Weight'],df3['Item_MRP'] , c=labels)
plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
sil_score = silhouette_score(new_df1, db.labels_)
db_score = davies_bouldin_score(new_df1, db.labels_)
ch_score = calinski_harabasz_score(new_df1, db.labels_)
print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)
print("Calinski-Harabasz Index:", ch_score)





