from jax import numpy as jnp
import os 
import pkg_resources
import numpy as np
from jax import jit
from jax import lax

package_root = pkg_resources.resource_filename("slampy", '')
script_path = os.path.realpath(__file__)
script_root = os.path.dirname(script_path)
project_path = os.path.join(script_root, "..", "..", "..", "..", "..", "..")

"""
node_start = np.array([
            self.node_id,
            word_level_int,
            self.word_id,
            parent_node_id,
            children_start_slicing_idx,
            children_end_slicing_idx,
        ]).astype(np.float64)
        #print("node_start.shape", node_start.shape)
        
        #print("node_feature.shape",self.node_feature.flatten().shape)
        node = np.hstack([node_start,self.node_feature.flatten()])
"""

def dynamic_slice_helper(model_gpu, start_index, end_index):
    return lax.dynamic_slice(model_gpu, (start_index, 0), (end_index - start_index + 1, model_gpu.shape[1]))

dynamic_slice_helper_jitted = jit(dynamic_slice_helper)

@jit
def find_word(model_gpu, entry_point_index):
    entry_point_mask = model_gpu[:, 0]==entry_point_index
    selected_indices = jnp.nonzero(entry_point_mask, size=1)[0]
    selected_row = model_gpu[selected_indices][0]
    children_start_index = lax.floor(selected_row[4]).astype(int)
    children_end_index = lax.floor(selected_row[5]).astype(int)
    
    #children = model_gpu[children_start_index:children_end_index, :]
    #children = lax.dynamic_slice(model_gpu, (children_start_index, 0), (children_end_index - children_start_index + 1, model_gpu.shape[1]))
    #children = dynamic_slice_helper(model_gpu, children_start_index, children_end_index)
    #children = model_gpu[children_start_index:children_end_index + 1, :]
    #children = lax.dynamic_slice(model_gpu, (children_start_index, 0), (children_end_index - children_start_index + 1, model_gpu.shape[1]))
    children = lax.dynamic_slice(model_gpu, (children_start_index, 0), (children_end_index - children_start_index + 1, model_gpu.shape[1]))

    return selected_row, children_start_index, children_end_index#, children# (children_start_index, children_end_index)#selected_row# entry_point#(children_start_index, children_end_index)
    # Find the indices where the mask is True
    #selected_indices = jnp.where(mask)[0]

    #children_start_index = selected_row[4]
    #children_end_index = selected_row[5]

    
    #return selected_row

class JaxVisionModel():
    
    def __init__(self,model_file_name):
        #voc_tree_train_results_compute_model_project_path_np = os.path.join(project_path, "slampy/visual_model/data/models/%s" % (model_file_name))
        voc_model_path = os.path.join(package_root,"visual_model/data/models/%s" % (model_file_name))
        self.model = np.load(voc_model_path)
        print("voc_model_path", voc_model_path)
        print("voc_model", self.model)
        
        # numpy test
        
        selected_row = self.model[self.model[:, 0] == 0.0][0]  
        
        print("ahh", selected_row)
        self.model = jnp.asarray(self.model)
        
        mask = self.model[:, 0] == 0.0
        print("mask outside jit", mask)
        
        mask = jnp.equal(self.model[:, 0], 0.0)
        print("mask outside jit", mask)
        
        print("self.model[0]", self.model[0])
        
        print("jnp.equal(model_gpu[:, 0], 0.0)", jnp.equal(self.model[:, 0], 0.0))
        
    
    def test(self, entry_point_index=0.0, des=None):
        print("entry_point_index", entry_point_index)
        word_entry_point = find_word(self.model, jnp.array([0.0])) # des, entry_point_index
        return word_entry_point

model_test = JaxVisionModel("voctree_train_results_k_12_d_350000_m_hamming_t_2024_02_04_21_20_14.pkl.3.compute.model.npy")

result = model_test.test()
print("model_test.test()", result)

print("model_test.test() 4", result[0][4])
print("model_test.test() 5", result[0][5])
