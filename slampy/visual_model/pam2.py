from numba import njit, prange, jit as nb_jit
import numpy as np
import numba as nb
from numba import threading_layer, get_num_threads
import math
from numba.types import ListType, int64, float64, Tuple, unicode_type, List, Buffer, float32, UniTuple

#nb.config.THREADING_LAYER = 'omp'

def euclidean_vector(stack1, stack2):
    return np.sqrt(((stack2 - stack1)**2).sum(axis=1))

#Buffer(float32, 2, 'C', True)
@nb_jit("Array(float32, 2, 'C'), UniTuple(int64,2), int64, unicode_type, ListType(int64)", nopython=True, nogil=True)
def kpp(data, data_shape, k, metric_str, seeds):
    ## a randomly selected centroid to the list
    if len(seeds):
        centroids = [i for i in seeds]    
    else:
        centroids = list(np.random.choice(data_shape[0], 1, replace=False))
    remaining = k - 1
    for _ in range(remaining):
        dist = [np.float32(x) for x in range(data_shape[0])]
        # find the distance from remaining points to the existing centroids and pick the minimum
        for i in prange(data_shape[0]):
            next = data[i, :] 
            d = np.inf
            for j in range(len(centroids)):
                temp_dist = 0.0
                if metric_str == "euclidean":
                    for k in range (data_shape[1]):
                        temp_dist += (next[k] - data[centroids[j]][k])**2
                    temp_dist = math.sqrt(temp_dist)
                elif metric_str == "hamming":
                    for k in range (data_shape[1]):
                        temp_dist += (next[k] != data[centroids[j]][k])
                d = np.minimum(d, temp_dist) 
            dist[i] = d#.append(d) 
        # select data point with maximum distance as our next centroid 
        dist = np.array(dist) 
        next_centroid = np.argmax(dist)
        centroids.append(next_centroid) 
        dist = [] 
    return centroids


@njit(parallel=True, nopython=True, nogil=True)# @nb_jit(nopython=True)
def get_permutation_cost_and_best_selection2(cluster, data):
    n_clusters = len(cluster)
    best_centroid_idx = 0
    sum_cost = 0
    costs = [0 for i in range(n_clusters)]
    best_centroid_idxs = [0 for i in range(n_clusters)]

    for cluster_point_id in prange(n_clusters):
        test_centroid = cluster[cluster_point_id]
        test_centroid_column = np.zeros(shape=(cluster.shape[0], data[test_centroid].shape[0]), dtype=data[test_centroid].dtype)
        test_centroid_column[:,:] = data[test_centroid].reshape(1, data[test_centroid].shape[0])     
        new_cluster_column = np.zeros(shape=(cluster.shape[0], data[test_centroid].shape[0]), dtype=data.dtype)
        for i in range(0, n_clusters):
            new_cluster_column[i] = data[cluster[i]]
        # np.sqrt(((new_cluster_column - test_centroid_column)**2).sum(axis=1)).sum()
        cost = np.sqrt(((new_cluster_column - test_centroid_column)**2).sum(axis=1)).sum()#np.sum(euclidean_vector(new_cluster_column, test_centroid_column))#np.sqrt(np.sum((new_cluster_column - test_centroid_column)**2))
        sum_cost = sum_cost + cost
        costs[cluster_point_id] = cost
        #best_centroid_idxs.append(cluster_point_id)
        best_centroid_idxs[cluster_point_id] = cluster_point_id
    
    lowest_code = np.argmin(np.asarray(costs))
    best_centroid_idx = best_centroid_idxs[lowest_code]
        
    return (best_centroid_idx, sum_cost)
"""
@njit(parallel=True)
def optimise_cluster_membership(data, data_shape, n, intital_cluster_indices): # data=visualFeatureVocabulary
    print("optimiseClusterMembership")
    #if intital_cluster_indices is None:
    #    index = np.random.choice(data.shape[0], n, replace=False) 
    #else:
    index = intital_cluster_indices
    centeroids = np.asarray([data[i] for i in index], dtype=data.dtype)# np.asarray(, dtype=data.dtype)
    centeroidVectors = [data]
    for centeroid in centeroids:
        centeroidColumn = np.zeros(shape=(data.shape[0],data.shape[1]), dtype=data.dtype)
        centeroidColumn[:,:] = centeroid
        centeroidVectors.append(centeroidColumn)
    stack = centeroidVectors# np.asarray(centeroidVectors, dtype=data.dtype)
    distances = np.zeros(shape=(n,1), dtype=data.dtype)# []
    for i in range(n): # 
        distances[i] = (np.sqrt(((stack[0] - stack[i+1])**2).sum(axis=1)))#euclidean_vector(stack[0], stack[i+1]))#metric(stack[0], stack[i+1])) .append
    distances = distances.T # np.asarray(distances, dtype=np.float32).T 
    closestClusterIndexVector = np.argmin(distances, axis=1)
    clusterInfo = []
    for i in range(n):
        #cluster = (np.where(np.equal(closestClusterIndexVector, i))[0])
        #cluster = np.delete(cluster, np.where(cluster==index[i]), 0)
        #clusterInfo.append(cluster)
        cluster = np.where(np.equal(closestClusterIndexVector, i))[0]
        mask = (cluster != int(index[i]))
        cluster2 = cluster[mask]
        clusterInfo.append(cluster2)
    clusteredPoints = []
    sumPoints = 0
    for i in range(n):
        clusteredPoints.append(clusterInfo[i])
        sumPoints = sumPoints + clusteredPoints[i].shape[0]
    assert sumPoints == data.shape[0] - n
    return (np.asarray([i for i in index]), clusteredPoints) #(np.asarray([i for i in index], dtype=np.int64), clusteredPoints)


#@njit(parallel=True)
def optimise_cluster_membership_newish(data, data_shape, n, intital_cluster_indices): # data=visualFeatureVocabulary
    #print("optimise_cluster_membership")
    index = intital_cluster_indices
    centeroids = np.asarray([data[i] for i in index], dtype=data.dtype)
    centroid_vectors = [data]
    for centroid in centeroids:
        #print("centroid", centroid)
        #centroid_column = np.zeros(shape=(data.shape[0],data.shape[1]), dtype=data.dtype)
        centroid_column = np.tile(centroid, (data.shape[0], 1)) #[:,:] = centroid ' (n, 1)
        centroid_vectors.append(centroid_column)
    stack = centroid_vectors # np.asarray(centroid_vectors, dtype=data.dtype)
    distances = []
    for i in range(n):
        first_stack = stack[0]
        second_stack = stack[i + 1]
        collect = []
        for j in range(first_stack.shape[0]):
            first_element = first_stack[j]
            second_element = second_stack[j]
            totaling = 0
            for k in range(first_element.shape[0]):
                totaling += (first_element[k] - second_element[k])**2
            collect.append(math.sqrt(totaling))
        distances.append(collect)#metric(stack[0], stack[i+1]))
    distances = np.asarray(distances, dtype=np.float32).T 
    closest_cluster_index_vector = np.argmin(distances, axis=1)
    clusterInfo = []
    for i in range(n):
        cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        #print("cluster!!1", cluster)
        #print("np.where(cluster==index[i])", np.where(cluster==index[i])[0])
        cluster = np.delete(cluster, np.where(cluster==index[i])[0], 0)
        clusterInfo.append(cluster)
    clusteredPoints = []
    sum_points = 0
    for i in range(n):
        clusteredPoints.append(clusterInfo[i])
        sum_points = sum_points + clusteredPoints[i].shape[0]
    assert sum_points == data.shape[0] - n
    return (np.asarray([i for i in index]), clusteredPoints) # , dtype=np.int64

"""
#@partial(jax.jit, static_argnums=(2,))

@njit(parallel=True, nopython=True, nogil=True)
def get_permutation_cost_and_best_selection(cluster, data):
    n_clusters = len(cluster)
    best_centroid_idx = 0
    sum_cost = 0
    costs = []
    best_centroid_idxs = []

    for cluster_point_id in prange(n_clusters):
        test_centroid = cluster[cluster_point_id]
        test_centroid_column = np.zeros(shape=(cluster.shape[0], data[test_centroid].shape[0]), dtype=data[test_centroid].dtype)
        test_centroid_column[:,:] = data[test_centroid].reshape(1, data[test_centroid].shape[0])      
        new_cluster_column_list = []
        for i in range(0, n_clusters):
            new_cluster_column_list.append(data[cluster[i]])
        #new_cluster_column = new_cluster_column_list#np.stack(new_cluster_column_list) 
        newClusterColumn = np.zeros(shape=(cluster.shape[0], data[test_centroid].shape[0]), dtype=data.dtype)
        for i in range(0, n_clusters):
            newClusterColumn[i] = data[cluster[i]]
        pairwise_distance = euclidean_vector(newClusterColumn, test_centroid_column) #(newClusterColumn - test_centroid_column)**2
        cost = np.sum(pairwise_distance)
        sum_cost = sum_cost + cost
        costs.append(cost)
        best_centroid_idxs.append(cluster_point_id)
    lowest_code = np.argmin(np.asarray(costs))
    best_centroid_idx = best_centroid_idxs[lowest_code]
    return (best_centroid_idx, sum_cost)

#@jax.jit
#@partial(jax.jit, static_argnums=(3,))
@njit(parallel=True, nopython=True, nogil=True)
def optimise_centroid_selection(centroids, clusters, data):
    new_centroid_ids = []
    new_clusters = []
    total_cost = 0
    total_centroids = len(centroids)
    
    #np.zeros(shape=(cluster.shape[0], data[test_centroid].shape[0]), dtype=data[test_centroid].dtype)
    new_centroid_ids = [0 for i in range(total_centroids)]
    new_clusters = [] # data_array_to_fill = np.zeros(shape=(len(clusters),data_shape[0]), dtype=nb.float32)

    for cluster_idx in range(len(clusters)):
        #print("cluster_idx", cluster_idx)
        # take current centroid from the data
        old_centroid = centroids[cluster_idx]
        # get cluster ids
        cluster_ids = clusters[cluster_idx] #np.asarray()
        #add old centroid into cluster stack
        full_stack = np.append(cluster_ids, old_centroid)
        #get best choice and get cost of arrangement
        print("about to do itttt....", full_stack.shape, full_stack, data)
        best_centroid_cluster_idx, sum_cost = get_permutation_cost_and_best_selection2(full_stack, data)
        total_cost = total_cost + sum_cost
        # remove best centroid from cluster
        new_centroid_ids[cluster_idx] = int(full_stack[best_centroid_cluster_idx])
        best_centroid_mask = np.where((full_stack==full_stack[best_centroid_cluster_idx]))[0]
        #cluster_stack_without_centroid = np.delete(full_stack, best_centroid_mask, 0)
        
        mask = (best_centroid_mask != int(full_stack[best_centroid_cluster_idx]))
        cluster_stack_without_centroid = best_centroid_mask[mask]
        #add new centroid to output
        #new_centroid_ids.append(int(full_stack[best_centroid_cluster_idx]))
        # add old centroid into cluster
        new_cluster_stack = np.append(old_centroid, cluster_stack_without_centroid)
        
        print("--------------------------------------------------------------")
        print(new_cluster_stack, len(new_cluster_stack))
        print("##################")
        new_clusters.append(new_cluster_stack) # [cluster_idx]
        #new_clusters.append(new_cluster_stack)
    return (new_centroid_ids, new_clusters, total_cost)



def optimise_cluster_membership_old(data, data_shape, n, intital_cluster_indices=None):
    print("data",data, intital_cluster_indices)
    #print("optimise_cluster_membership")
    if intital_cluster_indices is None:
        index = np.random.choice(data.shape[0], n, replace=False) 
    else:
        index = intital_cluster_indices
    centeroids = np.asarray([data[i] for i in index], dtype=data.dtype)
    centroid_vectors = [data]
    for centroid in centeroids:
        centeroidColumn = np.zeros(shape=(data.shape[0],data.shape[1]), dtype=data.dtype)
        centeroidColumn[:,:] = centroid
        centroid_vectors.append(centeroidColumn)
    stack = np.asarray(centroid_vectors, dtype=data.dtype)
    print("stack", len(stack), n)
    distances = []
    for i in range(n):
        distances.append(euclidean_vector(stack[0], stack[i+1])) # np.sqrt()
    distances = np.asarray(distances, dtype=np.float32).T 
    print("distances", distances.shape)
    closest_cluster_index_vector = np.argmin(distances, axis=1)
    print("closest_cluster_index_vector", closest_cluster_index_vector, closest_cluster_index_vector.shape)
    clusterInfo = []
    for i in range(n):
        cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        cluster2 = np.delete(cluster, np.where(cluster == int(index[i]))[0], 0)
        clusterInfo.append(cluster2)
    clusteredPoints = []
    sum_points = 0
    for i in range(n):
        clusteredPoints.append(clusterInfo[i])
        sum_points = sum_points + clusteredPoints[i].shape[0]
    print("data.shape",data.shape[0], sum_points)
    assert sum_points == data.shape[0] - n
    return (np.asarray([i for i in index]), clusteredPoints)


@njit(parallel=True, nopython=True, nogil=True)
def optimise_cluster_membership(data, data_shape, n, intital_cluster_indices): # data=visualFeatureVocabulary # =4 # =hamming_vector
    index = intital_cluster_indices
    centeroids = [data[i] for i in index]# data[index]
    centroid_vectors = [] # data
    data_array_to_fill = np.zeros(shape=(data_shape[0],data_shape[1]), dtype=nb.float32)
    for i in range(data_shape[0]):
        for k in range(data_shape[1]):
            data_array_to_fill[i][k] = data[i][k]
    centroid_vectors.append(data_array_to_fill)
    for centroid_idx in range(len(centeroids)):
        centroid = centeroids[centroid_idx]
        centeroidColumn = np.zeros(shape=(data_shape[0],data_shape[1]), dtype=nb.float32) # , dtype=data.dtype
        for j in range(data_shape[0]):
            for k in range(data_shape[1]):
                centeroidColumn[j, k] = centroid[k]
        centroid_vectors.append(centeroidColumn)
    stack = centroid_vectors
    d2 = np.zeros((n, stack[0].shape[0]), dtype=stack[0].dtype)
    for i in prange(n):
        first_stack = stack[0]
        second_stack = stack[i + 1]
        collect = []
        for j in range(first_stack.shape[0]):
            first_element = first_stack[j]
            second_element = second_stack[j]
            totaling = 0
            for k in range(first_element.shape[0]):
                totaling += (first_element[k] - second_element[k])**2
            collect.append(math.sqrt(totaling))
        d2[i] = collect   
    distances = np.asarray(d2, dtype=nb.float32).T
    closest_cluster_index_vector = np.argmin(distances, axis=1)
    clusterInfo = []
    for i in range(n):
        cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        mask = (cluster != int(index[i]))
        cluster2 = cluster[mask]
        clusterInfo.append(cluster2)
    clustered_points = []
    clustered_point_counts = []
    sum_points = 0
    for i in range(n):
        clustered_point_counts.append(len(clusterInfo[i]))
        clustered_points.append(clusterInfo[i])
        sum_points = sum_points + clustered_points[i].shape[0]
    assert sum_points == data_shape[0] - n
    return (np.asarray([i for i in index]), clustered_points)



def pam_fit(data, n_regions = 2, metric = "euclidean", seeding = "heuristic", seeds=None): #  seeds=nb.typed.List([0])
    # Get seeds
    print("numba.config.NUMBA_DEFAULT_NUM_THREADS", nb.config.NUMBA_DEFAULT_NUM_THREADS)
    
    print("numba.get_num_threads()", nb.get_num_threads())
    print("Threading layer chosen: %s" % threading_layer())
    
    #exit()
    print("Seeding centroids via heuristic.")
    if (seeding == "heuristic"):
        if seeds is None:
            random_choice = np.random.choice(data.shape[0], 1, replace=False) 
            seeds = nb.typed.List([int(random_choice)])
            print(data.shape, n_regions, metric, seeds)#
        ret_seeds = kpp(data, data.shape, n_regions, metric, seeds) # seeds
        seeds = np.asarray(ret_seeds)
    elif (seeding == "random"):
        seeds = np.random.choice(data, len(data), replace=False)
    else:
        seeds = seeding
        
    print("Seeding done.", seeds)
    
    last_centroids, last_clusters = optimise_cluster_membership(data, data.shape, n_regions, nb.typed.List(seeds))

    last_cost =  np.inf
    iteration = 0
    escape = False
    while not escape:
        iteration = iteration + 1
        print("Started Iteration %d. Performing optimise_centroid_selection" % iteration)
        current_centeroids, current_clusters, current_cost = optimise_centroid_selection(last_centroids, last_clusters, data)
        print("Continuing Iteration %d. Performing optimise_cluster_membership" % iteration) 
        current_centeroids, current_clusters = optimise_cluster_membership(data, data.shape, n_regions, nb.typed.List(list(current_centeroids)))
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


"""


def optimise_cluster_membership_old(data, data_shape, n, intital_cluster_indices=None):
    print("data",data, intital_cluster_indices)
    #print("optimise_cluster_membership")
    if intital_cluster_indices is None:
        index = np.random.choice(data.shape[0], n, replace=False) 
    else:
        index = intital_cluster_indices
    centeroids = np.asarray([data[i] for i in index], dtype=data.dtype)
    centroid_vectors = [data]
    for centroid in centeroids:
        centeroidColumn = np.zeros(shape=(data.shape[0],data.shape[1]), dtype=data.dtype)
        centeroidColumn[:,:] = centroid
        centroid_vectors.append(centeroidColumn)
    stack = np.asarray(centroid_vectors, dtype=data.dtype)
    print("stack", len(stack), n)
    distances = []
    for i in range(n):
        distances.append(euclidean_vector(stack[0], stack[i+1])) # np.sqrt()
    distances = np.asarray(distances, dtype=np.float32).T 
    print("distances", distances.shape)
    closest_cluster_index_vector = np.argmin(distances, axis=1)
    print("closest_cluster_index_vector", closest_cluster_index_vector, closest_cluster_index_vector.shape)
    clusterInfo = []
    for i in range(n):
        cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        cluster2 = np.delete(cluster, np.where(cluster == int(index[i]))[0], 0)
        clusterInfo.append(cluster2)
    clusteredPoints = []
    sum_points = 0
    for i in range(n):
        clusteredPoints.append(clusterInfo[i])
        sum_points = sum_points + clusteredPoints[i].shape[0]
    print("data.shape",data.shape[0], sum_points)
    assert sum_points == data.shape[0] - n
    return (np.asarray([i for i in index]), clusteredPoints)


@njit(parallel=True)
def optimise_cluster_membership_borked(data, data_shape, n, intital_cluster_indices): # data=visualFeatureVocabulary # =4 # =hamming_vector
    index = intital_cluster_indices
    centeroids = [data[i] for i in index]# data[index]
    centroid_vectors = [] # data
    data_array_to_fill = np.zeros(shape=(data_shape[0],data_shape[1]), dtype=nb.float32)
    for i in range(data_shape[0]):
        for k in range(data_shape[1]):
            data_array_to_fill[i][k] = data[i][k]
    centroid_vectors.append(data_array_to_fill)
    for centroid_idx in range(len(centeroids)):
        centroid = centeroids[centroid_idx]
        centeroidColumn = np.zeros(shape=(data_shape[0],data_shape[1]), dtype=nb.float32) # , dtype=data.dtype
        for j in range(data_shape[0]):
            for k in range(data_shape[1]):
                centeroidColumn[j, k] = centroid[k]
        centroid_vectors.append(centeroidColumn)
    stack = centroid_vectors
    d2 = np.zeros((n, stack[0].shape[0]), dtype=stack[0].dtype)
    for i in prange(n):
        first_stack = stack[0]
        second_stack = stack[i + 1]
        collect = []
        for j in range(first_stack.shape[0]):
            first_element = first_stack[j]
            second_element = second_stack[j]
            totaling = 0
            for k in range(first_element.shape[0]):
                totaling += (first_element[k] - second_element[k])**2
            collect.append(math.sqrt(totaling))
        d2[i] = collect   
    distances = np.asarray(d2, dtype=nb.float32).T
    closest_cluster_index_vector = np.argmin(distances, axis=1)
    clusterInfo = []
    for i in range(n):
        cluster = np.where(np.equal(closest_cluster_index_vector, i))[0]
        mask = (cluster != int(index[i]))
        cluster2 = cluster[mask]
        clusterInfo.append(cluster2)
    clustered_points = []
    clustered_point_counts = []
    sum_points = 0
    for i in range(n):
        clustered_point_counts.append(len(clusterInfo[i]))
        clustered_points.append(clusterInfo[i])
        sum_points = sum_points + clustered_points[i].shape[0]
    assert sum_points == data_shape[0] - n
    return (np.asarray([i for i in index]), clustered_points, clustered_point_counts)


"""