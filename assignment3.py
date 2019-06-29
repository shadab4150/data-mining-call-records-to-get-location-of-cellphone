import pandas as pd
from datetime import timedelta
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') 

def clusterInfo(model):
  print("Cluster Analysis Inertia: ", model.inertia_)
  print('------------------------------------------')
  for i in range(len(model.cluster_centers_)):
    print("\n  Cluster ", i)
    print("    Centroid ", model.cluster_centers_[i])
    print("    #Samples ", (model.labels_==i).sum()) # NumPy Power

# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
  # Ensure there's at least one cluster...
  minSamples = len(model.labels_)
  minCluster = 0
  for i in range(len(model.cluster_centers_)):
    if minSamples > (model.labels_==i).sum():
      minCluster = i
      minSamples = (model.labels_==i).sum()
  print("\n  Cluster With Fewest Samples: ", minCluster)
  return (model.labels_==minCluster)


def doKMeans(data, clusters=0):
 
  df1 = pd.concat([data.TowerLon, data.TowerLat], axis = 1)
  kmeans = KMeans(n_clusters = clusters)
  labels = kmeans.fit_predict(df1)
  
  centroids = kmeans.cluster_centers_
  ax.scatter(x = centroids[:, 0], y = centroids[:, 1], c = 'r', marker = 'x', s = 100)
  model = kmeans
  return model

df = pd.read_csv('F:\\CSV files\\CDR.csv')
print(df.head())
df.CallDate = pd.to_datetime(df.CallDate)
df.Duration = pd.to_timedelta(df.Duration)
df.CallTime = pd.to_timedelta(df.CallTime)
print(df.dtypes)
print(df[(df.TowerLat == 32.721986) & (df.TowerLon == -96.890587)]) #the data for second question (post office Lon/Lat))

users = df.In.unique()
print(users)

print("\n\nExamining person: ",6)
user1 = df[(df.In == users[ 6])]

user1 = user1[(user1.DOW == 'Mon') | (user1.DOW == 'Tue') | (user1.DOW == 'Wed') | (user1.DOW == 'Thu') | (user1.DOW == 'Fri')]

user1 = user1[(user1.CallTime < "17:00:00")]

fig=plt.figure()
ax = fig.add_subplot(111)
ax.scatter(user1.TowerLon,user1.TowerLat, c='g', marker='o', alpha=0.2)
ax.set_title('Weekday Calls before 5PM')

model = doKMeans(user1, 3)

midWayClusterIndices = clusterWithFewestSamples(model)
midWaySamples = user1[midWayClusterIndices]
print("    Its Waypoint Time: ", midWaySamples.CallTime.mean())

# visualize the results!
# First draw the X's for the clusters:
ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
ax.set_title('Weekday Calls Centroids')
plt.show()  
