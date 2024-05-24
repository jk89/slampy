import jax
from jax import random
from functools import partial
from numba import njit, prange, jit as nb_jit
import numpy as np
rng_key = random.PRNGKey(0)
import numba as nb

#distance metrics ######################

#vector metrics
def hamming_vector(stack1, stack2):
    return (stack1 != stack2).sum(axis=1)

def euclidean_vector(stack1, stack2):
    return np.sqrt(((stack2 - stack1)**2).sum(axis=1))
    #return (np.absolute(stack2-stack1)).sum(axis=1)

# point metrics
#@njit()
@nb_jit(nopython=True)
def euclidean_point(p1, p2): 
    #print("euclidean_point", p1, p2, p1.shape, p2.shape, p1.dtype)
    return np.sqrt(np.sum((p1 - p2)**2))

#@njit()
@nb_jit(nopython=True)
def hamming_point(p1, p2):
    return np.sum((p1 != p2))
import math
#@jax.jit
#@partial(jax.jit, static_argnums=(1,2))
#@njit(parallel=True)
#@njit(parallel=True)
@nb_jit(nopython=True)
def kpp(data, data_shape, k, metric_str): #metric=euclidean_point
    ## a randomly selected centroid to the list 
    centroids = list(np.random.choice(data_shape[0], 1, replace=False))#list(jax.random.choice(rng_key, data.shape[0], [1])) # , replace=False
    #print("centroids", centroids)
    #print("k", k)
    ## compute remaining k - 1 centroids
    remaining = k - 1
    #print("remainingremainingremaining", remaining)
    for c_id in range(remaining):
        #print("Selecting centroid", c_id)
        #dist = [] 
        dist = [np.float32(x) for x in range(0)]
        # find the distance from remaining points to the existing centroids and pick the minimum
        for i in range(data_shape[0]):
            #print("iii", i)
            #print("data", data, data.shape)
            next = data[i, :] 
            #print("nextnext", next)
            d = np.inf
            for j in range(len(centroids)):
                #print("centroids", centroids[0], len(centroids), j)
                #print("ahhhh", next.dtype, data[centroids[j]].dtype)
                temp_dist = 0.0
                if metric_str == "euclidean":
                    for k in range (data_shape[1]):
                        temp_dist += math.sqrt((next[k] - data[centroids[j]][k])**2)
                    #temp_dist = np.sum((next - data[centroids[j]])**2)#euclidean_point(next, data[centroids[j]]) 
                elif metric_str == "hamming":
                    for k in range (data_shape[1]):
                        temp_dist += (next[k] != data[centroids[j]][k])
                #    temp_dist = np.sum((next != data[centroids[j]]))#euclidean_point(next, data[centroids[j]]) 

                d = np.minimum(d, temp_dist) 
            dist.append(d) 
        # select data point with maximum distance as our next centroid 
        dist = np.array(dist) 
        next_centroid = np.argmax(dist) #data[np.argmax(dist), :] 
        centroids.append(next_centroid) 
        dist = [] 
    return centroids

#@jax.jit
#@partial(jax.jit, static_argnums=(2,))
def get_permutation_cost_and_best_selection(cluster, data, metric):
    n_clusters = len(cluster)
    best_centroid_idx = 0
    sum_cost = 0
    costs = []
    best_centroid_idxs = []

    for cluster_point_id in range(n_clusters):
        test_centroid = cluster[cluster_point_id]
        test_centroid_column = data[test_centroid].reshape(1, data[test_centroid].shape[0])
        test_centroid_column = np.zeros(shape=(cluster.shape[0], data[test_centroid].shape[0]), dtype=data[test_centroid].dtype)
        test_centroid_column[:,:] = data[test_centroid].reshape(1, data[test_centroid].shape[0])  
        new_cluster_column = np.zeros(shape=(cluster.shape[0], data[test_centroid].shape[0]), dtype=data.dtype)
        for i in range(0, n_clusters):
            new_cluster_column[i] = data[cluster[i]] # need to replace this line with something like a where clause???
        
        #new_cluster_column_list = []
        #for i in range(n_clusters):
        #    new_cluster_column_list.append(data[cluster[i]])

        #new_cluster_column = np.stack(new_cluster_column_list)
        
        #new_cluster_column = np.where(
        #    np.arange(n_clusters)[:, None] == np.arange(data[test_centroid].shape[0]),
        #    data[cluster],
        #    new_cluster_column
        #)[0]
        
        pairwise_distance = metric(new_cluster_column, test_centroid_column)
        cost = np.sum(pairwise_distance)
        sum_cost = sum_cost + cost
        costs.append(cost)
        best_centroid_idxs.append(cluster_point_id)
    
    lowest_code = np.argmin(np.asarray(costs))
    best_centroid_idx = best_centroid_idxs[lowest_code]
        
    return (best_centroid_idx, sum_cost)

#@jax.jit
#@partial(jax.jit, static_argnums=(3,))
def optimise_centroid_selection(centroids, clusters, data, metric):
    new_centroid_ids = []
    new_clusters = []
    total_cost = 0
    for cluster_idx in range(len(clusters)):
        #print("cluster_idx", cluster_idx)
        # take current centroid from the data
        old_centroid = centroids[cluster_idx]
        # get cluster ids
        cluster_ids = clusters[cluster_idx] #np.asarray()
        #add old centroid into cluster stack
        full_stack = np.append(cluster_ids, old_centroid)
        #get best choice and get cost of arrangement
        best_centroid_cluster_idx, sum_cost = get_permutation_cost_and_best_selection(full_stack, data, metric)
        total_cost = total_cost + sum_cost
        #add new centroid to output
        new_centroid_ids.append(int(full_stack[best_centroid_cluster_idx]))
        # remove best centroid from cluster
        best_centroid_mask = np.where((full_stack==full_stack[best_centroid_cluster_idx]))[0] #np.where(cluster_ids==best_centroid_cluster_idx)[0]
        cluster_stack_without_centroid = np.delete(full_stack, best_centroid_mask, 0)
        # add old centroid into cluster
        new_cluster_stack = np.append(old_centroid, cluster_stack_without_centroid)
        new_clusters.append(new_cluster_stack)
    return (new_centroid_ids, new_clusters, total_cost)

#@partial(jax.jit, static_argnums=(1,3))
#@nb_jit(nopython=True)
@njit(parallel=True)
def optimise_cluster_membership(data, data_shape, feature_length, n, intital_cluster_indices): # data=visualFeatureVocabulary # =4 # =hamming_vector
    #print("optimise_cluster_membership")
    index = intital_cluster_indices
    #centeroids = data[np.array(index, dtype=np.int64)] #data[np.array(index)]
    
    
    #centeroids = np.asarray([data[i] for i in index], dtype=data.dtype)
    centeroids = [data[i] for i in index]# data[index]

    centroid_vectors = [] # data
    
    data_array_to_fill = np.zeros(shape=(data_shape[0],data_shape[1]), dtype=nb.float32)
    for i in range(data_shape[0]):
        for k in range(data_shape[1]):
            data_array_to_fill[i][k] = data[i][k]
        #for k in range(data_shape[1]):
            

        # Append reconstructed non-read-only array to centroid_vectors
    centroid_vectors.append(data_array_to_fill) # data # .copy()
    
    for centroid_idx in range(len(centeroids)):
        centroid = centeroids[centroid_idx]
        #centroid.repeat(data_shape[0]).reshape((-1, data_shape[0]))
        centeroidColumn = np.zeros(shape=(data_shape[0],data_shape[1]), dtype=nb.float32) # , dtype=data.dtype
        for j in range(data_shape[0]):
            for k in range(data_shape[1]):
                centeroidColumn[j, k] = centroid[k]
        centroid_vectors.append(centeroidColumn)
        #centeroidColumn[:,:] = centroid
        #sub_stack = []
        #for element in np.asarray(centroid).shape:
        #    sub_stack.append(element)
        #centroid_vectors.append(sub_stack) #(centeroidColumn) # 
    

    stack = centroid_vectors # np.asarray(centroid_vectors, dtype=data.dtype)
    #distances = []
    

    
    #for i in range(n):
    #    print("stack 0", stack[0], stack[0].shape)
    #    print("metric(stack[0], stack[i+1])", metric(stack[0], stack[i+1]).shape)
    #    distances.append(metric(stack[0], stack[i+1])) # metric is like (np.absolute(stack2-stack1)).sum(axis=1)
    #distances = np.asarray(distances, dtype=np.float32).T # distances (770, 9)
    #
    """
    d2 = []
    for i in range(n):
        first_stack = stack[0]
        second_stack = stack[i+1]
        collect = []
        for j in range(first_stack.shape[0]):
            first_element = first_stack[j]
            second_element = second_stack[j]
            #print(first_element.shape, second_element)
            #collect.append(np.sqrt(sum((x - y) ** 2 for x, y in zip(first_element, second_element))))
            totaling = 0
            
            for k in range(first_element.shape[0]):
                totaling += abs(first_element[k] - second_element[k])
            
            collect.append(totaling)
            #collect.append(((sum(x - y) for x, y in zip(first_element, second_element))))
            #collect.append(euclidean_point(first_element, second_stack))
            #sum_elements = np.sum(first_element, second_element)
        d2.append(collect)
    """
    print("doinging some parrall stuff")
    d2 = np.zeros((n, stack[0].shape[0]), dtype=stack[0].dtype)
    for i in range(n):
        first_stack = stack[0]
        second_stack = stack[i + 1]
        collect = []

        for j in range(first_stack.shape[0]):
            first_element = first_stack[j]
            second_element = second_stack[j]

            totaling = 0
            for k in range(first_element.shape[0]):
                totaling += math.sqrt((first_element[k] - second_element[k])**2)

            collect.append(totaling)

        d2[i] = collect
    #print("d2", d2)
    
    distances = np.asarray(d2, dtype=nb.float32).T
    #print("distances", distances )
    #print("d2", d2)
            
    closest_cluster_index_vector = np.argmin(distances, axis=1)
    clusterInfo = []
    for i in range(n):
        #cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        #print("cluster!!1", cluster)
        #print("np.where(cluster==index[i])", np.where(cluster==index[i])[0])
        #cluster = np.delete(cluster, np.where(cluster==index[i])[0], 0)
        
        #cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        #cluster2 = np.delete(cluster, np.where(cluster == int(index[i]))[0], 0)
        #clusterInfo.append(cluster2)
    
        cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        mask = (cluster != int(index[i]))
        cluster2 = cluster[mask]
        clusterInfo.append(cluster2)
    
    clusteredPoints = []
    sum_points = 0
    for i in range(n):
        clusteredPoints.append(clusterInfo[i])
        sum_points = sum_points + clusteredPoints[i].shape[0]
    assert sum_points == data_shape[0] - n
    return (np.asarray([i for i in index]), clusteredPoints) # , dtype=np.int64


def pam_fit(data, n_regions = 2, metric = "euclidean", seeding = "heuristic"):
    if metric == "euclidean":
        point_metric = euclidean_point
        vector_metric = euclidean_vector
    elif metric == "hamming":
        point_metric = hamming_point
        vector_metric = hamming_vector
    else:
        #print("unsuported metric")
        return
    
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print("data", data, data.shape)
    # warm up jit stuff
    #jit_optimise_centroid_selection = lambda x: optimise_centroid_selection(x[0], x[1], x[2], vector_metric)

    # Get seeds
    print("Seeding centroids via heuristic.")
    if (seeding == "heuristic"):
        print("datttah", data, data.shape, data.dtype)
        ret_seeds = kpp(data, data.shape, n_regions, metric) # point_metric
        #print("ret_seeds", ret_seeds)
        seeds = np.asarray(ret_seeds)
    elif (seeding == "random"):
        seeds = np.random.choice(rng_key, data, len(data), replace=False)
    else:
        seeds = seeding
        
    print("Seeding done.")
    
    print("data", data)
    print("data.shape", data.shape)
    
    # Define a routine to keep going until cost stays the same or gets worse
    
    feature_length = 2
    
    last_centroids, last_clusters = optimise_cluster_membership(data, data.shape, feature_length, n_regions, nb.typed.List(seeds))
    
    print("Performed initial optimise_cluster_membership")

    last_cost =  np.inf
    iteration = 0
    escape = False
    while not escape:
        iteration = iteration + 1
        print("Started Iteration %d. Performing optimise_centroid_selection" % iteration)
        current_centeroids, current_clusters, current_cost = optimise_centroid_selection(last_centroids, last_clusters, data, vector_metric)
        print("Continuing Iteration %d. Performing optimise_cluster_membership" % iteration) 
        current_centeroids, current_clusters = optimise_cluster_membership(data, data.shape, feature_length, n_regions, nb.typed.List(list(current_centeroids)))
        #print((current_cost < last_cost, current_cost, last_cost, ))
        print("Current cost is less than last cost: %s. Current cost: %s, Last cost: %s, Change: %s" % (str(current_cost < last_cost),str(current_cost),str(last_cost),str(current_cost - last_cost)))
        if (current_cost<last_cost):
            print("Iteration %d completed. The cost is still improving, continuing with optimisation." % iteration, float(current_cost), float(last_cost))
            last_cost = current_cost
            last_centroids = current_centeroids
            last_clusters = current_clusters
        else:
            print("Iteration %d finished. The cost has either become worse or did not improve since the last iteration, abandoning optimisation." % iteration, current_cost, last_cost)
            escape = True
        #print("--------------------")
    return (last_centroids, last_clusters)