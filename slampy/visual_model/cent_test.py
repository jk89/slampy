import numba as nb
import numpy as np

@nb.jit(nopython=True)
def create_centroid_vectors(data, index):
    data_shape = data.shape
    centeroids = [data[i] for i in index]

    centroid_vectors = np.empty((len(centeroids)+1, data_shape[0], data_shape[1]))

    # Copy the original data
    centroid_vectors[0, :, :] = data

    # Copy the centroids
    for k in range(len(centeroids)):
        centroid_vectors[k+1, :, :] = centeroids[k]

    return centroid_vectors

# Example usage
data = np.random.rand(5, 3)
index = np.array([0, 2, 4])

result = create_centroid_vectors(data, index)