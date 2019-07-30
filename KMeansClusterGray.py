import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


# K Means Clustering Algorithm for Grayscale values

# Author: Jamie Walker, 29/07/2019
# Copyright © Jamie Walker 2019
# https://www.jamieleewalker.com

class KMeansClusterGray:
    """The K means clustering algorithm works by selecting k random values from the image, assigning each pixel in the
    image to a cluster, and then recalculating the cluster mean values based off the pixels in the cluster.
    It repeats this until convergence is reached or the maximum amount of iterations is reached.
    This implementation was actually made for images with a background of value 0. As such, it clusters the foreground
    and ignores the background. Therefore, it actually makes k + 1 clusters."""

    def __init__(self, input_filename: str, k: int):
        # K is the amount of clusters we want
        # However n is the real amount of clusters because we want to ignore 0 values
        self.k = k
        self.n = k + 1

        # Open the image and save relevant data
        self.input_filename = input_filename
        self.input_image = cv2.imread(self.input_filename, 0)
        self.image_shape = np.shape(self.input_image)

        print("Opened the {}x{} image: {}...".format(self.image_shape[0], self.image_shape[1], self.input_filename))

    # The main convert function
    # Returns the converted image and the cluster means
    def convert(self, n_max_iterations) -> (np.ndarray, np.ndarray):
        print("=========CONVERTING IMAGE=========")

        # Select K random values not equal to 0 from the image
        random_values = self.k_random_values()

        # Add the 0 value for the first cluster
        seed_values = np.append([0], random_values)
        print("Got seed values... {}".format(seed_values.tolist()))

        # Perform the iterations
        print("Beginning iterations...")
        (cluster_map, cluster_values) = self.k_means_iterate(seed_values, n_max_iterations)
        print("Iterations complete...")

        # Create an image from the cluster information
        converted_image = self.create_cluster_image(cluster_map, cluster_values)

        print("=========IMAGE  CONVERTED=========")
        return converted_image, cluster_values

    # Selects k random values not equal to 0 from the array
    def k_random_values(self) -> np.ndarray:
        # Preallocate array to return
        to_return = np.zeros((self.k, 1))

        # Generate k random values not equal to 0
        for i in range(self.k):
            value = 0

            # Keep generating until value is not 0 and not in the existing values
            while value == 0 or value in to_return:
                # Generate a random coordinate in the image
                rand_row = int(np.floor(random.random() * self.image_shape[0]))
                rand_col = int(np.floor(random.random() * self.image_shape[1]))

                # Get the value at that point
                value = self.input_image[rand_row, rand_col]

            # Add the value to the return array
            to_return[i] = value

        return to_return

    # Performs the iterations with the values
    def k_means_iterate(self, seed_values, n_max_iterations) -> (np.ndarray, np.ndarray):
        # Variables for the loop
        loop = True
        n_iterations = 0

        # The values to return
        cluster_map = self.assign_to_clusters(seed_values)
        cluster_values = seed_values

        # Old mean values to keep track of
        old_values = seed_values

        # Perform the iterations
        while loop:
            # Update the cluster information
            cluster_map = self.assign_to_clusters(cluster_values)
            cluster_values = self.update_means(cluster_map)

            # Check if there's no change
            if np.array_equal(old_values, cluster_values):
                # Stop the loop
                loop = False
            else:
                # Update the old values for the next iteration
                old_values = cluster_values

                # Increase the amount of iterations passed
                n_iterations = n_iterations + 1

                # Check if the maximum amount of iterations have been reached
                if n_iterations == n_max_iterations:
                    # Stop the loop
                    loop = False
                    print("WARNING: The maximum amount of iterations was reached before convergence was achieved")

        return cluster_map, cluster_values

    # Assigns each pixel to a cluster
    def assign_to_clusters(self, cluster_values) -> np.ndarray:
        # Create a row x col x n array for the differences in the values and current mean values
        differences = np.zeros((self.image_shape[0], self.image_shape[1], self.n))

        # Loop through each cluster
        for i in range(self.n):
            # Calculate the differences for each cluster value
            differences[:, :, i] = abs(self.input_image - cluster_values[i])

        # Make the difference for the 0 cluster very big so that it is ignored
        differences[:, :, 0] = 9223372036854775807

        # Create the cluster map by taking the minimum differences along the z axis
        cluster_map = differences.argmin(2)

        # Set the 0 values to the 0 cluster
        cluster_map[np.where(self.input_image == 0)] = 0

        return cluster_map

    # Updates each cluster mean value
    def update_means(self, cluster_map) -> np.ndarray:
        # Preallocate the array for the new mean values
        new_means = np.zeros((self.n, 1))

        # Loop through each cluster
        for i in range(self.n):
            # Calculate the new means from the image where each pixel pertains to the current cluster
            new_means[i] = np.mean(self.input_image[np.where(cluster_map == i)])

        return new_means

    # Creates the converted image based off the cluster map and cluster values
    def create_cluster_image(self, cluster_map, cluster_values) -> np.ndarray:
        # Preallocate the image array size
        image = np.zeros((self.image_shape[0], self.image_shape[1]))

        # Loop through each cluster
        for i in range(self.n):
            # Set the image pixels to the cluster mean where applicable
            image[np.where(cluster_map == i)] = cluster_values[i]

        return image


# TESTING FUNCTION
def test(file: str):
    clustering = KMeansClusterGray(file, 3)

    # Start by displaying the original image
    print("Showing original image...")
    plt.imshow(clustering.input_image, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # Convert the image...
    new_image = clustering.convert(200)

    # Display the converted image
    print("Showing converted image...")
    plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Author: Jamie Walker, 29/07/2019
# Copyright © Jamie Walker 2019
# https://www.jamieleewalker.com
