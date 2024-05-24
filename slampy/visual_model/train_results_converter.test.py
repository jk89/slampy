import unittest
from slampy.visual_model.train import VocTrain
from slampy.visual_model.train_results_converter import VocComputeModelGenerator

print("ahhh2")
class TrainResultsCoverter(unittest.TestCase):    
    def test_converted_results(self):
        self.assertTrue(True, f"Retrieved training results")
        trainer = VocTrain()
        trained_results = trainer.load_train_results_from_file("voctree_train_results_k_12_d_350000_m_hamming_t_2024_02_04_21_20_14.pkl")
        training_results_converter = VocComputeModelGenerator(trained_results)
        voc_vision_model = training_results_converter.get_compute_model()
        print("voc_vision_model", voc_vision_model)#
        training_results_converter.save_compute_model("voctree_train_results_k_12_d_350000_m_hamming_t_2024_02_04_21_20_14.pkl.compute.model.pkl", voc_vision_model)

if __name__ == '__main__':
    print("should be running test")
    unittest.main()