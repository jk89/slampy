import os 
import pkg_resources
import numpy as np
from numba import jit, njit, prange, types as nbtypes, typed as nbtyped, float32, float64

from slampy.visual_model.metrics import njit_hamming_vector
import timeit

package_root = pkg_resources.resource_filename("slampy", '')
script_path = os.path.realpath(__file__)
script_root = os.path.dirname(script_path)
project_path = os.path.join(script_root, "..", "..", "..", "..", "..", "..")

"""
[
    node_id,
    word_level_int,
    word_id,
    parent_node_id,
    children_start_slicing_idx,
    children_end_slicing_idx,
    <optional_word_weight> for a given model file this will either be there for the whole set or not at all.
    ...feature
]
"""

@njit()
def find_word(model, test_des, entry_point_index=0.0, train_format=False):
    #entry_point = model[model[:, 0] == entry_point_index][0] # FIXME Performance
    entry_point = model[int(entry_point_index)]
    children_start_index = int(entry_point[4])
    children_end_index = int(entry_point[5])
    #print("entry_point_index", entry_point_index)
    #print("children_start_index", children_start_index)
    #print("children_end_index", children_end_index)
    children = model[children_start_index:children_end_index+1, :]
    #print("children", children)
    #print("children.shape", children.shape)
    feature_offset_index = 7
    if train_format is True:
        feature_offset_index = 6
    #print("feature_offset_index", feature_offset_index)
    children_features = children[:,feature_offset_index:]
    #print(children_features.shape, test_des.shape, train_format)
    assert children_features.shape[1] == test_des.shape[0]
    #assert children_features.shape[0] == test_des.shape[0]
    #assert children_features.shape[1] == test_des.shape[1]
    test_stack = np.repeat(test_des, children_features.shape[0]).reshape(-1, children_features.shape[0]).T
    pairwise_distances = njit_hamming_vector(test_stack, children_features)
    #print("pairwise_distances.shape", pairwise_distances.shape)
    best_match = np.argmin(pairwise_distances)
    #print("best match")
    best_child = children[best_match]
    #print("children filter")
    best_child_is_word_level = best_child[1] == 1.0
    #print("best_child", best_child)    
    #print("best_child_is_word_level", best_child_is_word_level)
    if best_child_is_word_level is True:
        #print("return word")
        return best_child[2] # word id
    else:
        #print("recurse")
        best_child_index = best_child[0]
        return find_word(model, test_des, best_child_index, train_format)

@njit(parallel=True) # parallel=True
def find_words(model, test_des_stack, train_format=False):
    #print("train_format", train_format)
    n_test_stack = test_des_stack.shape[0]
    word_ids = np.zeros((1,n_test_stack),dtype=np.float32)
    #print("finding from n_test_stack words", n_test_stack)
    #print("test_des_stack.shape", test_des_stack.shape)
    for idx in prange(n_test_stack):
        #print("idx", idx, test_des_stack[idx].shape)
        new_word = find_word(model, test_des_stack[idx], train_format=train_format)
        #print("got new word",new_word,new_word)
        #print("word ids shape",word_ids.shape)
        word_ids[0, idx] = new_word
        #print("done idx", idx)
    return word_ids


"""
    node_id,
    word_level_int,
    word_id,
    parent_node_id,
"""
@njit()
def accend_node_id_ancestors(model, node_id, n_levels):
    # model[model[:, 0]
    
    #word_rows = model[model[:, 1] == 1.0]
    #print("node_id", node_id)
    
    current_node = model[int(node_id), :]# word_rows[word_rows[:,2]==word_id, :][0]
    #print("current word going to anscestors", current_node)
    for i in range(n_levels):
        #print("going up parents", n_levels, i)
        current_node_parent = current_node[3]
        #print("word parent", current_node_parent)
        current_node = model[int(current_node_parent),:] #word_rows[word_rows[:,2]==current_word_parent, :][0]
        #print("current word", current_word)
    return current_node[0]

"""
N rows with column:
    node_id,
    word_level_int,
    word_id,
    parent_node_id,
    children_start_slicing_idx,
    children_end_slicing_idx,
"""
#@njit() # model type is unknown N by X nd array, node_id is a float, and descendant_words is a list of floats

#@njit(nbtypes.ListType(float32)(float32[:,:], float32, nbtypes.ListType(float32)))

# , False, aligned=True


"""
0000000000000000000000000000000000000000000000000000000000000000000000
Entryyyy pointtt.... neeed to check for wwworrrddd [ 54.   1.   0.  37.   0.   0. 132. 174. 143.  31. 148. 251. 103. 186.
 246. 215.  70. 148. 227. 227. 107. 108.  81. 106.  42.  90.  54. 220.

"""

@njit(nbtypes.ListType(float32)(nbtypes.Array(float32, 2, 'C'), nbtypes.float32, nbtypes.Optional(nbtypes.ListType(float32))))
def find_descendant_words(model, node_id, descendant_words=None):
    if descendant_words is None:
        descendant_words = nbtyped.List.empty_list(float32)
    entry_point = model[int(node_id)]#model[model[:, 0] == node_id][0]
    
    #print("Entryyyy pointtt.... neeed to check for wwworrrddd", entry_point)
    entry_point_is_word = entry_point[1] == 1.0
    if entry_point_is_word is True:
        #print("entry_point_is_word")
        descendant_words.append(entry_point[2])
        return descendant_words
    #else:
        #print("entry_point is not word")
    
    children_start_index = int(entry_point[4])
    children_end_index = int(entry_point[5])
    children = model[children_start_index:children_end_index+1, :]
    
    #print("children!!!!!!!!!!!!!!!", len(children))
    for child in children:
        if child[1] == 1.0: # we are at word level
            #print("child is word! Adding to list", child, child[2])
            #if (child[0] == 54.0):
            #    print("------------------------------------------------ this one")
            descendant_words.append(child[2]) # add the word id
            #print("child[2]", child[2])
            #print("descendant_words", descendant_words)
        else:
            find_descendant_words(model, child[0], descendant_words) # keep going down
    descendant_words.sort()
    return descendant_words

@njit()
def compute_word_neighbourhood(model, word_id, n_levels):
    #FIXME this should be going from a word to a node id and then ascend
    #print("compute_word_neighbourhood levels", n_levels)
    words = model[model[:, 1] == 1.0]
    entry_point_word = words[words[:,2]==word_id][0]
    entry_point_word_node_id = entry_point_word[0]
    #accend_node_id_ancestors
    entry_point_word_ancestor_node_id = accend_node_id_ancestors(model, entry_point_word_node_id, n_levels)
    
    #print("entry_point_word_node_id", entry_point_word_node_id)
    #print("entry_point_word_ancestor_node_id", entry_point_word_ancestor_node_id)
    
    common_descendant_words = find_descendant_words(model, entry_point_word_ancestor_node_id, None)
    return common_descendant_words

@jit(parallel=True)
def compute_vocab_words_neighbourhood(model, n_levels):
    words = model[model[:, 1] == 1.0]
    empty_list_of_floats = nbtyped.List.empty_list(nbtypes.float32)
    vocab_neighbourhood_ids = [empty_list_of_floats.copy() for _ in range(len(words))]
    for i in prange(len(words)):
        word = words[i]
        word_neighbourhood = compute_word_neighbourhood(model, word[2], n_levels)
        vocab_neighbourhood_ids[i] = word_neighbourhood
        #for item in word_neighbourhood:
        #    vocab_neighbourhood_ids[i].append(item) # this works when parallel is false
        #vocab_neighbourhood_ids[i].extend(word_neighbourhood) # this works to when parallel is false
    return vocab_neighbourhood_ids

class NumbaVisionModel():
    
    def __init__(self,model_file_name):
        voc_model_path = os.path.join(package_root,"visual_model/data/models/%s" % (model_file_name))
        self.model = np.load(voc_model_path)
        #print("voc_model_path", voc_model_path)
        #print("voc_model", self.model)
        
    def test(self, test_des, train_format=False, entry_point_index=0.0):
        word_entry_point = find_word(self.model, test_des, entry_point_index, train_format=train_format)
        return word_entry_point

def simple_test():
    start_time = timeit.default_timer()
    vision_model = NumbaVisionModel("voctree_train_results_k_12_d_350000_m_hamming_t_2024_02_04_21_20_14.pkl.5.compute.model.npy")
    end_time = timeit.default_timer()
    print("Load model took %s [s]", str(end_time-start_time))
    
    random_row_index = np.random.choice(vision_model.model.shape[0])
    test_des = vision_model.model[random_row_index][6:]
    
    start_time = timeit.default_timer()
    result = vision_model.test(test_des, True)
    end_time = timeit.default_timer()
    print("First result", result, " time taken %s [s]" % (str(end_time-start_time)))
    
    random_row_index = np.random.choice(vision_model.model.shape[0])
    test_des = vision_model.model[random_row_index][6:]

    start_time = timeit.default_timer()
    result = vision_model.test(test_des, True)
    end_time = timeit.default_timer()
    print("Second result", result, " time taken %s [s]" % (str(end_time-start_time)))
    
    random_row_index = np.random.choice(vision_model.model.shape[0])
    test_des = vision_model.model[random_row_index][6:]

    start_time = timeit.default_timer()
    result = vision_model.test(test_des, True)
    end_time = timeit.default_timer()
    print("Third result", result, " time taken %s [s]" % (str(end_time-start_time)))
    
    print("Neighbourhood test") 
    
    #Need to patch the word ids because they are broken in this model
    
    model32 = vision_model.model.astype(np.float32)
    
    word_indices = np.where(model32[:, 1] == 1)[0]
    # Generate new unique word ids for each word
    new_word_ids = np.arange(len(word_indices))
    # Update the third column (index 2) of the selected words with the new word ids
    model32[word_indices, 2] = new_word_ids
        
    # word zero node id
    word_rows = model32[model32[:, 1] == 1]
    first_word_node_id = word_rows[word_rows[:,2]==0.0, :][0][0]
    print("first_word_node_id", first_word_node_id)
    
    
    #print("ancestor-----------",  first_word_node_id, 0, accend_node_id_ancestors(model32, first_word_node_id, 0))
    #print("ancestor1-----------",  first_word_node_id, 1, accend_node_id_ancestors(model32, first_word_node_id, 1))
    #print("ancestor2-----------",  first_word_node_id, 2, accend_node_id_ancestors(model32, first_word_node_id, 2))
    #print("ancestor3-----------",  first_word_node_id, 3, accend_node_id_ancestors(model32, first_word_node_id, 3))
    #print("ancestor4-----------",  first_word_node_id, 4, accend_node_id_ancestors(model32, first_word_node_id, 4))
    #print("ancestor5-----------",  first_word_node_id, 5, accend_node_id_ancestors(model32, first_word_node_id, 5))
    #print("ancestor6-----------",  first_word_node_id, 6, accend_node_id_ancestors(model32, first_word_node_id, 6))
    depth = 1
    first_ancestor_node_id =accend_node_id_ancestors(model32, first_word_node_id, depth)
    print("first_ancestor_node_id", first_ancestor_node_id)
    
    print("0000000000000000000000000000000000000000000000000000000000000000000000")
    

    
    common_descendant_words = find_descendant_words(model32, first_ancestor_node_id, None)
    
    
    #exit()

    common_words = compute_word_neighbourhood(vision_model.model.astype(np.float32), 0.0, depth)
    print("common_words", len(common_words), common_words)
    print("common_descendant_words", len(common_descendant_words), common_descendant_words)
    
    all_vocab_neighbourhood = compute_vocab_words_neighbourhood(model32, depth)
    #all_vocab_neighbourhood =[list(i) for i in all_vocab_neighbourhood]
    print("all_vocab_neighbourhood", all_vocab_neighbourhood, all_vocab_neighbourhood.shape)

if __name__ == "__main__":
    simple_test()