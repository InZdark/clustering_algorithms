import numpy as np

# distance function
def Distance(x, means, *args,**kwargs):
  diff=[]
  dists=[]
  for mean in means:
    diff= x-mean
    dist = np.sqrt(np.sum(diff**2,axis=1,keepdims=True))
    dists.append(dist)
  return np.hstack(dists)


class Kmeans:
    ''' K Means Algorithm (iteractive update k centroids)
            1. initialize k centroids
            2. calculate the distance between each data point and k centroids
            3. assgin the data to a centroid if the distance is the least
            4. update centroid
            5. repeat 1 ~ 4
    '''

    def __init__(self, k, distance_func=Distance):
        ''' Description:
            - k: number of centroids
            - distance_func: you can customize your distance function
        '''
        self.k = k
        self.distance_func = distance_func

    def fit(self, X, iterations = 5):
        ''' - Description:
                - Fit X into k centroids, each centroid is a cluster.
            - Input:
                - X: data
            - Return:
                - self.means: k centroids
        '''
        indices = np.arange(X.shape[0])
        # randomly pick k data points as initialized centroids.
        centroid_locs = np.random.choice(indices, size=self.k, replace=False)
        self.means = X[centroid_locs]

        for i in range(iterations):
            y_hat = self.predict(X)
            self.means = []
            for j in range(self.k):
                mean = np.mean(X[y_hat == j], axis = 0)
                self.means.append(mean)
            self.means = np.vstack(self.means)
        return self.means

    def predict(self, X):
        dist = self.distance_func(X, self.means)
        y_hat = np.argmin(dist, axis = 1)
        return y_hat
