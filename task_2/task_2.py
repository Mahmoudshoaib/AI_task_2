import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Wuzzuf_Jobs.csv')
# Q-1
z = dataset.iloc[:, 5:9]

z['fact_of_YearsExp'] = pd.factorize(z['YearsExp'])[0]
# Q-2

dataset["Title"]= pd.factorize(dataset["Title"])[0]
dataset["Company"]= pd.factorize(dataset["Company"])[0]
y = dataset.iloc[:, [0,1]].values

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 31):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 31), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(y)
y_kmeans == 0

plt.scatter(y[y_kmeans == 0, 0], y[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(y[y_kmeans == 1, 0], y[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(y[y_kmeans == 2, 0], y[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(y[y_kmeans == 3, 0], y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(y[y_kmeans == 4, 0], y[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of job titles')
plt.xlabel('jobs')
plt.ylabel('Companies')
plt.legend()
plt.show()
