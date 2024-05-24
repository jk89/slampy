from slampy.visual_model.train import VocTrain
import unittest
import numpy as np


class TestPam(unittest.TestCase):    
    #def test_train(self):
    #    voc_trainer = VocTrain()
    #    voc_bundle_sample = voc_trainer.get_orb_training_sample(n=1000)
    #    trained_voc_tree = voc_trainer.train(voc_bundle_sample, 12)
    #    self.assertTrue(True, f"FTraining of voctree completed.")
    def test_train_hard(self):
        voc_trainer = VocTrain()
        voc_bundle_sample = voc_trainer.get_orb_training_sample(n=350000)
        trained_voc_tree = voc_trainer.train(voc_bundle_sample, 12, "hamming")
        self.assertTrue(True, f"FTraining of voctree hard completed.")
        

if __name__ == '__main__':
    unittest.main()