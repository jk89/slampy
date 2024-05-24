import unittest
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from jax import numpy as np
from pam import pam_fit

class TestPam(unittest.TestCase):    
    def test_sphere(self):
        sample = read_sample(FCPS_SAMPLES.SAMPLE_TARGET) 
        data = np.asarray(sample, dtype=np.float32)
        print("data", data)
        regions = 9
        self.assertTrue(True, f"Failed for by ref check 1")
        centroids, clusters = pam_fit(np.asarray(data, dtype=np.float32), 9)
        print("centroids", centroids)
        print("clusters", clusters)
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, sample)
        visualizer.show()

if __name__ == '__main__':
    unittest.main()