"""
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

parent is the parent layer node
level data is the centroids data do the metric comparison to find the child idx to recurse
children is a list of layer nodes
worldLayer indicates whether or not we are a end

What structure do we want for our vocmodel:

Essentially an ndarray where we have the following columns for each row
node_id, word_level_boolean, word_id, parent_id, children_start_slicing_idx, children_end_slicing_idx, feature_vector # node_id=0 identifies the root node and this does not
actually have a feature vector, it is just an entry point to the tree of feature vectors.
we first have the root node and then it specifies its children slicing indicies... for these children we can compare our incoming
feature vector to test the children vector and find the closest one. we then take its children and continue recursing.
We can us continuous blocks of children to make it easy.

"""

from slampy.visual_model.train import VocTrainLayer, VocTrain
from slampy.visual_model.model_numba import find_word, compute_vocab_words_neighbourhood, find_words
from typing import List
import numpy as np
import os 
import pkg_resources
import pickle

package_root = pkg_resources.resource_filename("slampy", '')
script_path = os.path.realpath(__file__)
script_root = os.path.dirname(script_path)
project_path = os.path.join(script_root, "..", "..", "..", "..", "..", "..")

# vstack

class UniqueIdGenerator():
    current_idx = 0
    def __init__(self):
        self.current_idx = 0
    def get(self):
        latest_idx = self.current_idx
        self.current_idx += 1
        return latest_idx

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

class VocComputeNode():
    #node_id = None
    #word_id = None
    #node_feature = None
    #word_layer_bool = False
    #children: List["VocComputeNode"] = []
    #parent:"VocComputeNode" = None
    def get_compute_node(self):
        n_children = len(self.children)
        if self.word_layer_bool is True:
            children_start_slicing_idx = 0
            children_end_slicing_idx = 0
            
        else:
            children_start_slicing_idx = self.children[0].node_id
            children_end_slicing_idx = self.children[n_children-1].node_id
        parent_node_id = 0
        if self.parent is not None:
            parent_node_id = self.parent.node_id
        word_level_int = int(self.word_layer_bool)
        #print("node_feature.shape", self.node_feature.shape)
        node_start = np.array([
            self.node_id,
            word_level_int,
            self.word_id,
            parent_node_id,
            children_start_slicing_idx,
            children_end_slicing_idx,
        ]).astype(np.float32)
        node = np.hstack([node_start,self.node_feature.flatten()])
        return node

    def __init__(self, node_id, word_layer_bool, word_id, parent_compute_node, node_feature):
        self.node_id = node_id
        self.word_id = word_id
        self.word_layer_bool = word_layer_bool
        self.parent = parent_compute_node
        self.node_feature = node_feature
        self.children = []


class VocComputeModelGenerator():
    unique_node_id_generator: UniqueIdGenerator = None
    unique_word_id_generator: UniqueIdGenerator = None

    def __init__(self, voc_train_model_level:VocTrainLayer):
        self.voc_train_model_level = voc_train_model_level
        self.unique_node_id_generator = UniqueIdGenerator()
        self.unique_word_id_generator = UniqueIdGenerator()

    def handle_train_level(self,train_level:VocTrainLayer, parent_compute_node:VocComputeNode):
        n_children = 0
        if train_level.children is not None:
            n_children = len(train_level.children)
        current_level_are_words = train_level.wordLayer
        for child_idx in range(n_children):
            child_feature = train_level.levelData[child_idx]
            # FIXME sometimes we dont set the word_ids correctly too many appear 0 when they should have a value. Mitigated below.
            if current_level_are_words is True:
                word_id = self.unique_word_id_generator.get()
            else:
                word_id = 0
            is_word_layer = train_level.children[child_idx].wordLayer
            child_compute_node = VocComputeNode(
                self.unique_node_id_generator.get(),
                is_word_layer, 
                word_id,
                parent_compute_node,
                child_feature
            )
            parent_compute_node.children.append(child_compute_node)        
        if n_children == 0:
            for child_feature in train_level.levelData:
                word_id = self.unique_word_id_generator.get()
                is_word_layer = True
                child_compute_node = VocComputeNode(
                    self.unique_node_id_generator.get(),
                    is_word_layer, 
                    word_id,
                    parent_compute_node,
                    child_feature
                )
                parent_compute_node.children.append(child_compute_node)   
        # now that we have this levels compute nodes we can recurse and do the same for the children.
        if train_level.wordLayer is False:
            for child_idx in range(n_children):
                 child_compute_node = parent_compute_node.children[child_idx]
                 next_train_level = train_level.children[child_idx]
                 self.handle_train_level(next_train_level, child_compute_node)
        return parent_compute_node

    def pack_compute_model(self, parent_compute_node: VocComputeNode, compute_node_stack=[]):
        top_level = False
        if len(compute_node_stack) == 0:
            compute_node_flat = parent_compute_node.get_compute_node()
            compute_node_stack.append(compute_node_flat)
            top_level = True
        n_children = len(parent_compute_node.children)
        for child_idx in range(n_children):
            child_compute_node = parent_compute_node.children[child_idx]
            child_compute_node_flat = child_compute_node.get_compute_node()
            compute_node_stack.append(child_compute_node_flat)
        
        for child_idx in range(n_children):
            self.pack_compute_model(parent_compute_node.children[child_idx], compute_node_stack)
        
        if top_level is True:
            return np.vstack(compute_node_stack)
        return None

    def get_handle_model(self):
        root_train_level: VocTrainLayer = self.voc_train_model_level
        data_type = root_train_level.levelData.dtype
        self.feature_size = root_train_level.levelData.shape[1]
        dummy_root_feature = np.zeros((1,self.feature_size)).astype(data_type)
        root_compute_node = VocComputeNode(self.unique_node_id_generator.get(), False, 0, None, dummy_root_feature)
        return self.handle_train_level(root_train_level,root_compute_node)

    def get_trained_model_weight(self, compute_model, total_number_of_words):
        # get voc training data sample
        training_data = VocTrain()
        vocabulary = training_data.get_orb_training_bundle()
        weights = {}
        for i in range(total_number_of_words):
            weights[i] = 0
        total = 0
        words = find_words(compute_model, vocabulary, train_format=True)[0]
        print("words.shape", words, words.shape)
        for word in words:
            print("word", word)
            total += 1
            wordId = word
            weights[int(wordId)] += 1
        #for test_des in vocabulary:
        #    total += 1
        #    wordId = find_word(compute_model, test_des, train_format=True)
        #    print("wordId in get weights", wordId)
        #    weights[wordId] += 1
        for key in weights.keys():
            weights[key] = weights[key] / total
        return weights

    def get_compute_model(self, n_neighbourhood_levels=1):
        root_train_level: VocTrainLayer = self.voc_train_model_level
        data_type = root_train_level.levelData.dtype
        self.feature_size = root_train_level.levelData.shape[1]
        dummy_root_feature = np.zeros((1,self.feature_size)).astype(data_type)
        root_compute_node = VocComputeNode(self.unique_node_id_generator.get(), False, 0, None, dummy_root_feature)
        print("Unpacking trained tree")
        root_compute_node = self.handle_train_level(root_train_level,root_compute_node)
        print("Collecting packed compute model")
        packed_vision_model = self.pack_compute_model(root_compute_node)
        print("Forging word indicies")
        #FIXME the words are borked for some reason and i cant be bothered anymore so will just fix it here
        word_indices = np.where(packed_vision_model[:, 1] == 1.0)[0]
        # Generate new unique word ids for each word
        new_word_ids = np.arange(len(word_indices))
        # Update the third column (index 2) of the selected words with the new word ids
        packed_vision_model[word_indices, 2] = new_word_ids
        total_number_of_words = len(word_indices)
        print("Getting trained model weights")
        print("This could take a while!")
        word_weights = self.get_trained_model_weight(packed_vision_model, total_number_of_words)
        word_weights_stack = []
        for i in range(total_number_of_words):
            word_weights_stack.append(word_weights[i])
            print("word_weights[i]", i, word_weights[i])
        word_weights_stack = np.asarray(word_weights_stack, dtype=np.float32)
        print("word_weights_stack.shape",word_weights_stack.shape)
        print("Computing word neighbourood")
        vocab_neighbourhood = compute_vocab_words_neighbourhood(packed_vision_model, n_neighbourhood_levels)
        native_vocab_neighbourhood = [list(i) for i in vocab_neighbourhood]
        #inject weights
        #print("Inserting model word weights") # failed here last time...
        #vision_model_with_weights = np.insert(packed_vision_model, 6, word_weights_stack, axis=1)
        print("Returning final model")
        final_model = {"model":packed_vision_model,"vocab_neighbourhood": native_vocab_neighbourhood, "n_words": total_number_of_words, "word_weights": word_weights_stack}
        print("final_model", final_model)
        #final_model = {"model":vision_model_with_weights,"vocab_neighbourhood": vocab_neighbourhood, "n_words": total_number_of_words, "word_weights": word_weights_stack}
        return final_model    
    """
        node_id,
        word_level_int,
        word_id,
        parent_node_id,
    """    
    
    def save_compute_model(self, file_name, model):
        voc_tree_train_results_compute_model_project_path_np = os.path.join(project_path, "slampy/visual_model/data/models/%s" % (file_name))
        voc_tree_train_results_compute_model_package_path_np = os.path.join(package_root,"visual_model/data/models/%s" % (file_name))
        with open(voc_tree_train_results_compute_model_package_path_np, "wb") as fout1:
            pickle.dump(model,fout1)
        with open(voc_tree_train_results_compute_model_project_path_np, "wb") as fout2:
            pickle.dump(model,fout2)
        #np.save(voc_tree_train_results_compute_model_package_path_np, model)
        #np.save(voc_tree_train_results_compute_model_project_path_np, model)
        

#reminder of the class from train.py
"""

import numpy as np
from metrics import hammingVector

def compareLevelData(levelData, vec, metric=hammingVector):
    lenLevelData = levelData.shape[0]
    #print(levelData, levelData.shape)
    lenFeature = levelData.shape[1]
    testColumn = np.zeros(shape=(lenLevelData, lenFeature), dtype=levelData.dtype)
    for i in range(0, lenLevelData):
        testColumn[i] = vec
    pairwiseDistance =  metric(levelData, vec)
    minIndex = np.argmin(pairwiseDistance)
    return int(np.argmin(pairwiseDistance))

def findWordInternal(_layers, vec, path=[], level=-1):
    levelData = _layers.levelData
    closestChild = compareLevelData(levelData, vec)
    path.append(closestChild)
    closestChildLayer = _layers.children[closestChild]
    if type(closestChildLayer) is tuple:
        closestChildLayer = closestChildLayer[0]
    if closestChildLayer.wordLayer:
        #print("llll", closestChildLayer.levelData)
        if closestChildLayer.levelData.shape[0] != 0:
            closestGrandChild = compareLevelData(closestChildLayer.levelData, vec)
            path.append(closestGrandChild)
            return path
        else:
            return path
    else:
        return findWordInternal(closestChildLayer, vec, path)

    
def findWord(_layers, vec):
    returnData = findWordInternal(_layers, vec, [])
    return "-".join([str(i) for i in returnData])

def traverseAllWords(_layers, above=[], mapp={}, Z=0):
    if _layers.children is not None:
        #iterate children
        children = []
        for i in range(len(_layers.children)):
            if type(_layers.children[i]) is tuple:
                child = _layers.children[i][0] # traverseAllWords(_layers.children[i][0])
            else:
                child = _layers.children[i] # traverseAllWords()
            children.append(child)
        #print(Z, (children), above)
        if len(children) == 0:
            paths = [above + [i] for i in range(_layers.levelData.shape[0])]
            for path in paths:
                key = "-".join([str(j) for j in path])
                mapp[key] = True
            return
        for i in range(len(children)):
            Y = Z + 1
            traverseAllWords(children[i], above + [i], mapp, Y)
    else:
        paths = [above + [i] for i in range(_layers.levelData.shape[0])]
        #print(Z, "no Children", paths, above, _layers.levelData)
        if _layers.levelData.shape[0] != 0:
            for path in paths:
                key = "-".join([str(j) for j in path])
                mapp[key] = True
        else:
            mapp["-".join([str(j) for j in above])] = True
    outputMap = {}
    dictKeys = list(mapp.keys())
    for i in range(len(dictKeys)):
        outputMap[dictKeys[i]] = i
    return outputMap

def WordToVec(vec, _wordToVecMap, _layers, bowSize, weights=None):
    wordId = findWord(_layers, vec)
    #print(wordId)
    vecId = _wordToVecMap[wordId]
    #print(vecId)
    feature = np.zeros(shape=(1, bowSize), dtype=vec.dtype)
    #print(feature.shape)
    #weight!
    if weights is not None:
        weight = 1
        if wordId in weights:
            weight = weights[wordId]            
        feature[0][vecId] = weight   
    else:
        feature[0][vecId] = 1
    return feature

#get phrase weight over whole data set
def getWordWeights(_data, _layers):
    weights = {}
    total = 0
    for point in _data:
        total += 1
        wordId = findWord(_layers, point)
        #print(wordId)
        if wordId not in weights:
            weights[wordId] = 1
        else:
            weights[wordId] = weights[wordId] + 1
    #print(total, weights)
    for key in weights.keys():
        weights[key] = weights[key] / total
    return weights

def frameTOBowVec(des1, bowSize, layers, wordToVecMap):
    bowVec = np.zeros(shape=(1, bowSize), dtype=des1.dtype)
    for i in range(des1.shape[0]):
        wordVec = WordToVec(des1[i], wordToVecMap, layers, bowSize)
        bowVec  = bowVec + wordVec
    #bowVec = bowVec / des1.shape[0]
    #normalise
    sumBow = np.sum(bowVec)
    bowVec = bowVec / sumBow
    return bowVec

def getLevelDict(_layers, outputMap = {}, above=[]):
    if len(above) == 0:
        outputMap["root"] = _layers.levelData.tolist()
    if _layers.children is not None:
        children = []
        lenChildren = len(_layers.children)
        for i in range(lenChildren):
            if type(_layers.children[i]) is tuple:
                child = _layers.children[i][0]
            else:
                child = _layers.children[i]
            children.append(child)
        if lenChildren == 0: #does nothing
            for i in range(_layers.levelData.shape[0]):
                path = above + [i]
                key = "-".join([str(j) for j in path])
                outputMap[key] = children[i].levelData.tolist()
            return
        for i in range(lenChildren):
            path = above + [i]
            key = "-".join([str(j) for j in path])
            outputMap[key] = children[i].levelData.tolist()
            getLevelDict(children[i], outputMap, above + [i])
    else:
        outputMap["-".join([str(j) for j in above])] = _layers.levelData.tolist()
    return outputMap

def generateChildrenIdMap(_layers, parent=None, outputMap={}):
    if _layers.children is not None:
        children = []
        lenChildren = len(_layers.children)
        for i in range(lenChildren):
            if type(_layers.children[i]) is tuple:
                child = _layers.children[i][0] # traverseAllWords(_layers.children[i][0])
            else:
                child = _layers.children[i] # traverseAllWords()
            children.append(child)
        if parent is None:
            outputMap["root"] = [str(i) for i in range(lenChildren)]
            for i in range(lenChildren):
                generateChildrenIdMap(children[i], [i], outputMap)
        else:
            # parent  [0, 0, 3]
            myLevelKey = "-".join([str(j) for j in parent])
            outputMap[myLevelKey] = []
            for i in range(lenChildren):
                childPath = parent + [i]
                childKey = "-".join([str(j) for j in childPath])
                outputMap[myLevelKey].append(childKey)
                generateChildrenIdMap(children[i], [i], outputMap)
    return outputMap
"""
"""
def get_model():
    model = {
        "data": levelDataDict,
        "children": childMap, # depricate
        "wordIndex": wordToVecMap,
        "wordWeights": weights
    }

import pickle
with open('visionModelK4N50k-2.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

