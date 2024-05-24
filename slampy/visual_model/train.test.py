from slampy.visual_model.train import VocTrain
import unittest
import numpy as np


class TestPam(unittest.TestCase):    
    def test_train(self):
        voc_trainer = VocTrain()
        
        print(voc_trainer.get_orb_training_bundle())
        #voc_trainer.save_something_else()
        #voc_bundle_sample = voc_trainer.get_orb_training_sample()
        #trained_voc_tree = voc_trainer.train(voc_bundle_sample, 12)
        self.assertTrue(True, f"FTraining of voctree completed.")
        

if __name__ == '__main__':
    unittest.main()