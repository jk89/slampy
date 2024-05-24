import numpy as np
import os 
from random import sample
import cv2
from slampy.visual_model.pam2 import pam_fit
import pkg_resources
from datetime import datetime
import pickle

package_root = pkg_resources.resource_filename("slampy", '')
script_path = os.path.realpath(__file__)
script_root = os.path.dirname(script_path)
project_path = os.path.join(script_root, "..", "..", "..", "..", "..")

orb_training_bundle_path = os.path.join(package_root,"visual_model/data/orb_training_bundle.yml")
training_bundle_project_path_np = os.path.join(project_path, "visual_model/data/orb_training_bundle")
training_bundle_package_path_np = os.path.join(package_root,"visual_model/data/orb_training_bundle")

class VocTrainLayer:
    levelData = None
    children = []
    parent = None
    wordLayer = False
    def __init__(self, levelData, wordLayer = False, children=None, parent=None):
        self.parent = parent
        self.wordLayer = wordLayer
        self.children = children
        self.levelData = np.asarray(levelData)

class VocTrain():
    def __init__(self):
        self.__load_orb_training_bundle__()
    def __load_orb_training_bundle__(self):
        try:
            self.orb_training_bundle = np.load(training_bundle_package_path_np)
            return self.orb_training_bundle
        except Exception as e:
            print(f"Error: Unable to open the orb_training_bundle YAML file at {orb_training_bundle_path}", e)
            
        #ob = cv2.FileStorage(orb_training_bundle_path, cv2.FILE_STORAGE_READ)
        #if ob.isOpened():
        #    # Read data from the file
        #    print("Loading orb_training_bundle")
        #    data = ob.getNode("orb_training_bundle").mat()
        #    self.orb_training_bundle = data
        #    ob.release()
        #    print("Loading orb_training_bundle finished")
        #    return self.orb_training_bundle
        
    def get_orb_training_bundle(self):
        return self.orb_training_bundle
    def get_orb_training_sample(self,n=100000):
        random_indices = np.random.randint(0, n, size=n)
        return self.orb_training_bundle[random_indices]
   # def write_visual_model(self, visual_model_path):
   #     ob = cv2.FileStorage(visual_model_path, cv2.FileStorage_WRITE)
   #     ob.write("orb_training_bundle", visual_model_path)
   #     ob.release()
    def createLayer(self, centroidIndices, clustersIndicies, data, parent=None, k=110, metric="hamming"):
        centroidData = np.asarray([data[i] for i in centroidIndices])
        children = []
        parentLayer = VocTrainLayer(centroidData, False, children, parent)
        for i in range(len(centroidIndices)):
            clusterIndices = clustersIndicies[i]
            #global_clusterIndices = clusterIndices
            clusterData = [data[i] for i in clusterIndices]
            lenCluster = len(clusterIndices)
            #print("lenCluster, k", lenCluster, k)
            if lenCluster > k:
                # we should fit again
                childCenteroids, childClusters = pam_fit(clusterData, k, metric)
                clusterLayer = VocTrainLayer(childCenteroids, childClusters, clusterData, parentLayer, k, metric),
                children.append(clusterLayer)
            else:
                clusterLayer = VocTrainLayer(clusterData, True, None, parentLayer)
                children.append(clusterLayer)
        parentLayer.children = children
        return parentLayer
    def train(self, data, k=4, metric="hamming"):
        bestCentroidsORB, bestClustersORB = pam_fit(data, k, metric)
        training_results = self.createLayer( bestCentroidsORB, bestClustersORB, data, None, k, metric)
        self.training_results = training_results
        print("training_results", training_results)
        #os.path.join(project_path, 
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = "voctree_train_results_k_%s_d_%s_m_%s_t_%s.pkl" % (k, len(data), metric, formatted_datetime)
        voc_tree_train_results_project_path_np = os.path.join(project_path, "visual_model/data/%s" % (file_name))
        voc_tree_train_results_package_path_np = os.path.join(package_root,"visual_model/data/%s" % (file_name))
        with open(voc_tree_train_results_project_path_np, 'wb') as file_1:
            with open(voc_tree_train_results_package_path_np, 'wb') as file_2:
                pickle.dump(training_results, file_1)
                pickle.dump(training_results, file_2)
        #np.save(voc_tree_train_results_project_path_np, )
        return training_results
        
    #def save_something_else(self):
    #    training_sample = self.get_orb_training_sample()
    #    print("saving", training_bundle_package_path_np)
    #    np.save(training_bundle_package_path_np, training_sample)
    #    print("saving", training_bundle_project_path_np)
    #    np.save(training_bundle_project_path_np, training_sample)
    #    print("save complete")

#vt = VocTrain()
#orb_training_bundle = vt.get_orb_training_bundle()
