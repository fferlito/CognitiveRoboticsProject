import numpy as np


class OnlineKMeans:
    """ Online K Means Algorithm """

    def __init__(self,
                 num_features: int,
                 num_clusters: int,
                 lr: tuple = None):
        """
        :param num_features: The dimension of the data
        :param num_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param lr: The learning rate of the online k-means (c', t0). If None, then we will use the simplest update
        rule (c'=1, t0=0) as described in the lecture.
        """
        if num_features < 1:
            raise ValueError(f"num_features must be greater or equal to 1!\nGet {num_features}")
        if num_clusters < 1:
            raise ValueError(f"num_clusters must be greater or equal to 1!\nGet {num_clusters}")

        self.num_features = num_features
        self.num_clusters = num_clusters

        self.num_centroids = 0
        self.centroid = np.zeros((num_clusters, num_features))
        self.cluster_counter = np.zeros(num_clusters)  # Count how many points have been assigned into this cluster

        self.num_samples = 0
        self.lr = lr

    def fit(self, X):
        """
        Receive a sample (or mini batch of samples) online, and update the centroids of the clusters
        :param X: (num_features,) or (num_samples, num_features)
        :return:
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        for i in range(num_samples):
            self.num_samples += 1
            # Did not find enough samples, directly set it to mean
            if self.num_centroids < self.num_clusters:
                self.centroid[self.num_centroids] = X[i]
                self.cluster_counter[self.num_centroids] += 1
                self.num_centroids += 1
            else:
                # Determine the closest centroid for this sample
                sample = X[i]
                dist = np.linalg.norm(self.centroid - sample, axis=1)
                centroid_idx = np.argmin(dist)

                if self.lr is None:
                    self.centroid[centroid_idx] = (self.cluster_counter[centroid_idx] * self.centroid[centroid_idx] +
                                                   sample) / (self.cluster_counter[centroid_idx] + 1)
                    self.cluster_counter[centroid_idx] += 1
                else:
                    c_prime, t0 = self.lr
                    rate = c_prime / (t0 + self.num_samples)
                    self.centroid[centroid_idx] = (1 - rate) * self.centroid[centroid_idx] + rate * sample
                    self.cluster_counter[centroid_idx] += 1

    def predict(self, X):
        """
        Predict the cluster labels for each sample in X
        :param X: (num_features,) or (num_samples, num_features)
        :return: Returned index starts from zero
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        clusters = np.zeros(num_samples)
        for i in range(num_samples):
            sample = X[i]
            dist = np.linalg.norm(self.centroid - sample, axis=1)
            clusters[i] = np.argmin(dist)
        return clusters

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param X: (num_features,) or (num_samples, num_features)
        :return:
        """
        # Because the centroid may change in the online setting, we cannot determine the cluster of each label until
        # we finish fitting.
        self.fit(X)
        return self.predict(X)

    def calculate_cost(self, X):
        """
        Calculate the KMean cost on the dataset X
        The cost is defined in the L2 distance.

        :param X: (num_features,) or (num_samples, num_features) the dataset
        :return: The cost of this KMean
        """

        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        cost = 0
        for i in range(num_samples):
            # Determine the closest centroid for this sample
            sample = X[i]
            dist = np.linalg.norm(self.centroid - sample, axis=1)
            cost += np.square(np.min(dist))

        return cost


class DynamicOnlineKmeans():
    """ An implementation of the https://arxiv.org/abs/1412.5721"""

    def __init__(self,
                 num_features: int,
                 num_clusters: int, ):
        if num_features < 1:
            raise ValueError(f"num_features must be greater or equal to 1!\nGet {num_features}")
        if num_clusters < 1:
            raise ValueError(f"num_clusters must be greater or equal to 1!\nGet {num_clusters}")

        # Start from k+1 clusters
        self.k = num_clusters  # Need this to initialize the f_rc
        num_clusters = num_clusters + 1

        self.num_features = num_features
        self.num_clusters = num_clusters

        self.num_centroids = 0
        self.centroid = np.zeros((num_clusters, num_features))

        self.q = 0
        self.num_samples = 0
        self.f_rc = 0

    def fit(self, X):
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        for i in range(num_samples):
            self.num_samples += 1
            # Did not find enough samples, directly set it to mean
            if self.num_centroids < self.num_clusters:
                self.centroid[self.num_centroids] = X[i]
                self.num_centroids += 1

                if self.num_centroids == self.num_clusters:
                    # Initialize f_rc
                    min_dist_list = []
                    for c in range(self.num_clusters):
                        if c + 1 == self.num_clusters:
                            break
                        temp = self.centroid - self.centroid[c]
                        dist = np.sum(np.square(temp[c + 1:]), axis=1)
                        min_dist_list.append(np.min(dist))
                    w_star = np.min(min_dist_list) / 2

                    self.f_rc = w_star / self.k
            else:
                # Determine the closest centroid for this sample
                sample = X[i]
                dist = np.linalg.norm(self.centroid - sample, axis=1)
                centroid_idx = np.argmin(dist)

                prob_cup = min(np.square(dist[centroid_idx]) / self.f_rc, 1)

                if np.random.rand(1) < prob_cup:
                    # Expand the centroid
                    self.centroid = np.vstack((self.centroid, sample))
                    self.num_centroids += 1
                    self.q += 1

                if self.q > 3 * self.k * (1 + np.log(self.num_samples)):
                    self.q = 0
                    self.f_rc = 2 * self.f_rc

    def predict(self, X):
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        clusters = np.zeros(num_samples)
        for i in range(num_samples):
            sample = X[i]
            dist = np.linalg.norm(self.centroid - sample, axis=1)
            clusters[i] = np.argmin(dist)
        return clusters

    def calculate_cost(self, X):
        """
        Calculate the KMean cost on the dataset X
        The cost is defined in the L2 distance.

        :param X: (num_features,) or (num_samples, num_features) the dataset
        :return: The cost of this KMean
        """

        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        cost = 0
        for i in range(num_samples):
            # Determine the closest centroid for this sample
            sample = X[i]
            dist = np.linalg.norm(self.centroid - sample, axis=1)
            cost += np.square(np.min(dist))

        return cost
