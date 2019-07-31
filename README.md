# Jamie Walker - Python

This repository is just a collection of my python code that I've developed. 
Feel free to use any of it if it may be helpful for you...

## KMeansClusterGray.py
This is an implementation of the K-Means clustering algorithm I made
for grayscale images. It was initially made for grayscale images with
backgrounds of all 0 values. It ignores the background and
therefore it actually makes k+1 clusters.

#### Documentation / How to use
Create a new instance of the KMeansClusterGray
class, passing in the input filename and the amount of clusters to
the constructor.

`clustering = KMeansClusterGray(filename, 3)`

Call the KMeansClusterGray.convert(n_max_iterations) function.

The function takes in the maximum number of iterations to perform.
It returns a numpy array of the clustered image and a numpy
array containing the mean values for the clusters.

`clustered_image, cluster_values = clustering.convert(200)`