import numpy as np

class KMeansAlgorithm:
    def __init__(self, n_clusters, max_iters):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def initialize_centroids(self, points):
        """
        Description
        -----------
            Randomly initialize the centroids from the data points.

        Parameters
        ----
            points (np.ndarray): Data points.

        Returns
        -------
            np.ndarray: Initial k centroids.
        """
        # Get random index
        rand_indices = np.random.choice(points.shape[0], size = self.n_clusters, replace = False)
        # Select the points corresponding to these ind. as the iniital centroids
        self.centroids = points[rand_indices, :]

    def calculate_distance(self, points):
        """
        Description
        -----------
            Compute the euclidean distance between data points and centroids.

        Parameters
        ----
            points (np.ndarray): Data points.

        Returns
        -------
            np.ndarray: Distances between data points and centroids.
        """
        # Get matrix with zeros (0)
        distances = np.zeros((points.shape[0], self.centroids.shape[0]))

        # Iteration through each point and centroid and ccalculate the euclidean distance
        for i, point in enumerate(points):
            for j, centroid in enumerate(self.centroids):
                distances[i, j] = np.linalg.norm(point - centroid)

        return distances
    
    def closest_centroid(self, distances):
        """
        Description
        -----------
            Obtain the index of the closest centroid, minimum distance, for each data point.

        Parameters
        ----
            distances (np.ndarray): Distances between data points and centroids.

        Returns
        -------
            np.ndarray: Index of the closest centroid for each data point.
        """
        closest_centroid_indices = np.argmin(distances, axis = 1)

        return closest_centroid_indices
    
    def update_centroids(self, points, closest):
        """
        Description
        -----------
            Update centroid positions as the mean of all assigned points.

        Parameters
        ----
            points (np.ndarray): Data points.
            closest (np.ndarray): Index of the closest centroid for each data point.

        Returns
        -------
            np.ndarray: Updated centroids.
        """
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.centroids.shape[0]):

            # Get points in the centriud
            assigned_points = points[closest == i]
            # If the centroid has poiints
            if len(assigned_points) > 0:
                # Calculate the new centroid the new centroid as the AVG. of the assigned points
                new_centroids[i] = np.mean(assigned_points, axis=0)
            else:
                # In case the centroidee has no points, it is randomly relocated
                new_centroids[i] = points[np.random.choice(range(len(points)))] 

        self.centroids = new_centroids

    def fit(self, points):
        """
        Description
        -----------
        Repeat the process for a maximum number of iterations.

        Parameters
        ----------
            points (np.ndarray): Data points.
¿
        Returns
        -------
            np.ndarray: Final centroids.
        """

        self.initialize_centroids(points)
        for _ in range(self.max_iters):
            distances = self.calculate_distance(points)
            closest = self.closest_centroid(distances)
            self.update_centroids(points, closest)

        return self.centroids, closest