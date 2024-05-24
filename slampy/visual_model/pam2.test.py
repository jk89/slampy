import unittest
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.samples.definitions import SIMPLE_SAMPLES
import numpy as np
from pam2 import pam_fit

class TestPam2(unittest.TestCase):    
    def test_sphere(self):
        sample = read_sample(FCPS_SAMPLES.SAMPLE_GOLF_BALL) #SAMPLE_TARGET 
        data = np.asarray(sample, dtype=np.float32)
        regions = 9
        self.assertTrue(True, f"Failed for by ref check 1")
        centroids, clusters = pam_fit(np.asarray(data, dtype=np.float32), 5)
        print("centroids", centroids)
        print("clusters", clusters)
        print("centroid members", [len(i) for i in clusters])
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, sample)
        visualizer.show()

if __name__ == '__main__':
    unittest.main()