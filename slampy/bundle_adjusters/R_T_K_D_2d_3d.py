"""
Copyright Jonathan Kelsey 2023. All rights reserved.s

The "loss_function" params are split into two conceptual groups:

"params_to_optimise" aka params we have to optimise via the loss_function result minimisation and we have "params_static" which
represent fixed information in the system, than should not evolve.

"params_to_optimise" set (needs to be flattened into a 1 dimensional buffer)

- "map_points_3d" a list of 3d points in world space. We have M of these so, we have a Mx3 matrix.
- "camera_intrinsic_params" for the N cameras, we have 5 (variable) K values and 5 D values, so in total we have (N x 10) of theses
- "camera_frame_extrinsic_params" this is more complicated as we have B_i frames (each a different number) for each camera
    and these frames have a 3 rvecs and 3 tvecs. So here we have 6 values per frame and B_i frames per camera and there are N cameras, 
    this adds some complexity as we need to pack/unpack this array. We will have sum(6 x B_i for i in range(N)) values in total, 
    it makes sense as this is an array and not a fixed size matrix for all cameras to organise this by the "fixed" per invocation number
    of cameras N and then to have B_i chunks of these the data objects (6) organised as blocks and rely on "params_static" "camera_frame_metadata"
    in order to know the size of each of these blocks per camera frame, for unpacking and repacking purposes.
- "camera_frame_map_points_2d" The 2d points per frame for each frame per camera for all cameras.
- "camera_frame_map_points_2d" we have, B_i frames for each ith camera, for each of those B_i camera frames we have L_i,j 2d points Lx2.
    A total of sum(2 x L_i,j x B_i  for i,k in B_i for i in range(N)). So this is a nested set of lists, from the highest level
    we can index by the camera index, then next level have another array for each of the frames of a given camera, and within that list
    is our 2d points.
- "camera_frame_map_points_2d_3d_index" same shape as camera_frame_map_points_2d however we have a single value for each camera frame
    2d point representing the index of where you could find the corresponding "map_point_3d" for this 2d camera frame feature,
    a total of sum(1 x L_i,j x B_i  for i,k in B_i for i in range(N))

"params_static" set (also needs to be flattened into a 1 dimensional buffer)

- "n_cameras" The number of cameras N
- "all_camera_frame_counts" The number of frames per N camera  (N x 1)
- "all_camera_frame_point_counts" The number of points that varies for each frame for each camera for all cameras.
- "camera_frame_map_points_2d_3d_index" same shape as camera_frame_map_points_2d however we have a single value for each camera frame
    2d point representing the index of where you could find the corresponding "map_point_3d" for this 2d camera frame feature,
    a total of sum(1 x L_i,j x B_i  for i,k in B_i for i in range(N))
 
##############################################################################################################################################

# Packing ------------------------------------------------------------------------------------------------------------------------------------

We start with the array version of the data as input:

"map_points_3d" = [[1,1,1],[1,1,1]] as a (M x 3)
"camera_intrinsic_params_Ks" = [K_1,K_2....K_N] for N cameras each K is a (3 x 3)
"camera_intrinsic_params_Ds" = [D_1,D_2....D_N] for N cameras each D is a 5,
"camera_frame_extrinsic_Rs" = [[camera_1_frame_1_R, camera_1_frame_2_R...], [camera_2_frame_1_R, camera_2_frame_2_R...]] and each R is a 3x3
"camera_frame_extrinsic_Ts" = [[camera_1_frame_1_T, camera_1_frame_2_T...], [camera_2_frame_1_T, camera_2_frame_2_T...]] and each T is 3,
"camera_frame_map_points_2d" = [[[camera_1_frame_1_points_2d],[camera_1_frame_2_points_2d],...],[[camera_2_frame_1_points_2d],[camera_1_frame_2_points_2d],...],...]
the shape of this is of the form of a list of vector stacks which are the 2d points for each given camera frame. 

# Packing "params_to_optimise" ---------------------------------------------------------------------------------------------------------------

Before we pack the params_to_optimise, we need to minimise the data we are transmitting, the K, R information is in 3x3 
as this is the right form to do computations with. But in reality we only need 3 rvec values and 5 K values, and we dont want the other information
to be modified, such as the 1's and 0's values in K, which should be fixed and might throw off the optimiser.

camera_frame_extrinsic_rvecs = [ cv2.Rodriguez(R_per_frame_per_camera) for R_per_frame_per_camera in Rs_per_frame_per_camera for Rs_per_frame_per_camera in camera_frame_extrinsic_Rs ]
camera_intrinsic_params_Ks = [ [K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]]] for K in camera_intrinsic_params_Ks ]

Flattening of the "params_to_optimise" 1d array:

params_to_optimise = [map_points_3d.flatten(), camera_intrinsic_params_Ks.flatten(), camera_intrinsic_params_Ds.flatten(), camera_frame_extrinsic_Rs.flatten(), camera_frame_extrinsic_Ts.flatten()]

Packing "params_static" ---------------------------------------------------------------------------------------------------------------------

"n_cameras" the number of cameras.
"all_camera_frame_counts" The number of frames per N camera (N x 1)
"all_camera_frame_point_counts" The number of points that varies for each frame for each camera for all cameras.
"camera_frame_map_points_2d" = [[[camera_1_frame_1_points_2d],[camera_1_frame_2_points_2d],...],[[camera_2_frame_1_points_2d],[camera_1_frame_2_points_2d],...],...]
the shape of this is of the form of a list of vector stacks which are the 2d points for each given camera frame for all frames for all cameras.
"camera_frame_map_points_2d_3d_index" is of the form [[camera_1_frame_1_point_1_2d_3d_index, camera_1_frame_1_point_2_2d_3d_index,..],[camera_1_frame_2_point_1_2d_3d_index, camera_1_frame_2_point_2_2d_3d_index,..],[[camera_2_frame_1_point_1_2d_3d_index, camera_2_frame_1_point_2_2d_3d_index,..],[camera_2_frame_2_point_1_2d_3d_index, camera_2_frame_2_point_2_2d_3d_index],...],...]
and is of the shape of the form of a list of lists of lists each containing a point_2d_3d index for a given point for a given frame for a given camera for all cameras.

Total number of cameras:

This can be done by lookin at either the shape of the K's or the D's

n_cameras = len(camera_intrinsic_params_Ks)

Number of frames per camera:

This can be done from looking at the shape of either the R's or T's, each "camera_frame_extrinsic_Rs" is organised first by camera
and then by frames, so the frames per camera is just

all_camera_frame_counts = list(map(lambda x: len x), [camera_R_poses for camera_R_poses in camera_frame_extrinsic_Rs])

Number of points per frame per camera:

all_camera_frame_point_counts=[]
for camera in camera_frame_map_points_2d:
    all_camera_frame_point_counts=[]
    for frame in camera:
        frame_point_counts = []
        for points_mtx in frame:
            frame_point_counts.append(points_mtx.shape[0])
        all_camera_frame_point_counts.append(frame_point_counts)

all_camera_frame_point_counts =  all_camera_frame_point_counts

"params_static" = [n_cameras, all_camera_frame_counts.flatten(), all_camera_frame_point_counts.flatten(), camera_frame_map_points_2d.flatten(),camera_frame_map_points_2d_3d_index.flatten()]

# Unpacking -------------------------------------------------------------------------------------------------------------------------------------

# Unpack "params_static" (Tricky!) -----------------------------------------------------------------------------------------------------------------------

We start with "params_static" 1D array and "params_to_optimise" 1D array and we need to reconstruct our vectors and matricies.

Firstly work with the "params_static" 1D array and get the unnpacking information.

# Number of cameras

n_cameras = params_static[0]

# Number of frames per camera for all cameras

all_camera_frame_counts = params_static[1:n_cameras] # The number of frames per N camera (N x 1) e.g. [3,3] if we had 2 cameras each with 3 images.

# Number of points per camera frame per camera for all cameras

So from all_camera_frame_counts we know from our example(only an example frame counts are not expected to be the same) that with 2 cameras 
with 3 images we had all_camera_frame_count=[3,3] all_camera_frame_point_counts was constructed to give us the number of points per frame 
per camera for all cameras and there are 6 of these.

total_number_of_frames = sum(all_camera_frame_counts)
all_camera_frame_point_counts_flat = params_static[n_cameras+1,n_cameras+1+total_number_of_frames]

for our all_camera_frame_counts=[3,3] example then for the all_camera_frame_point_counts structure we would have one point count number
for each of the entries in [3,3] so we would have a shape like [[1,2,3],[4,5,6]] which would mean we have for camera 1, 3 frames with, frame 1 with
1 camera_frame_point, for frame 2, we would have 2 camera_frame_points and for frame 3, we would have 3 camera_frame_point, for camera 2 we have
also got 3 frames but this time we have 4 point in frame1, 5 points for frame 2 and 6 points for frame 3. So what is the shape we need to recover?

we currently have [1,2,3,4,5,6] and we know we have 2 cameras, the number of frames we have for each camera is all_camera_frame_counts[0] aka 3 
and all_camera_frame_counts[1] also happens to be 3. So we need to decompose the frames to their respective camers.

all_camera_frame_counts = [5, 1]  # Example with two cameras, camera 1 has 5 frames, and camera 2 has 1 frame
all_camera_frame_point_counts_flat = np.array([6, 5, 4, 3, 2, 1])  # Flattened array of point counts for the individual frames

# Calculate starting indices for each camera's data
start_indices = [0]
for frame_count in all_camera_frame_counts:
    start_indices.append(start_indices[-1] + frame_count)

# Reshape the data into a list of lists, where each inner list represents a camera
all_camera_frame_point_count = []
for i in range(len(start_indices) - 1):
    start = start_indices[i]
    end = start_indices[i + 1]
    camera_frame_point_count = all_camera_frame_point_counts_flat[start:end]
    all_camera_frame_point_count.append(camera_frame_point_count.tolist())

print(all_camera_frame_point_count)  # Output: [[6, 5, 4, 3, 2], [1]] # So we have for camera 1: frame 1 has 6 points, frame 2 has 5 etc...
and for camera 2: we have a single frame with 1 point.

Next recover camera_frame_map_points_2d and camera_frame_map_points_2d_3d_index

start_indices = ...  # previously calculated

total_number_of_2d_points = np.sum(all_camera_frame_point_count)  # Calculate total number of 2D points
camera_frame_map_points_2d_flat = params_static[start_indices[-1] : start_indices[-1] + total_number_of_2d_points]
camera_frame_map_points_2d_3d_index_flat = params_static[start_indices[-1] + total_number_of_2d_points : start_indices[-1] + 2 * total_number_of_2d_points]

camera_frame_map_points_2d = []
camera_frame_map_points_2d_3d_index = []

for i in range(len(start_indices) - 1):
    start = start_indices[i]
    end = start_indices[i + 1]
    camera_points_2d = []
    camera_points_2d_3d_index = []

    # Accumulate points for each frame within this camera
    current_point_index = start  # Track starting index for each frame
    for frame_point_count in all_camera_frame_point_count[i]:  # Use point counts for the current camera
        frame_points_2d = camera_frame_map_points_2d_flat[current_point_index : current_point_index + frame_point_count]
        frame_points_2d_3d_index = camera_frame_map_points_2d_3d_index_flat[current_point_index : current_point_index + frame_point_count]

        frame_points_2d = frame_points_2d.reshape((-1, 2))  # Reshape to (N, 2) for 2D points

        camera_points_2d.append(frame_points_2d)
        camera_points_2d_3d_index.append(frame_points_2d_3d_index.tolist())

        current_point_index += frame_point_count  # Move to the next frame's starting index

    camera_frame_map_points_2d.append(camera_points_2d)
    camera_frame_map_points_2d_3d_index.append(camera_points_2d_3d_index)
    
# Unpack "params_to_optimise" (Tricky!) -----------------------------------------------------------------------------------------------------------------------

    "map_points_3d" = [[1,1,1],[1,1,1]] as a (M x 3)
    "camera_intrinsic_params_Ks" = [K_1,K_2....K_N] for N cameras each K is a (3 x 3)
    "camera_intrinsic_params_Ds" = [D_1,D_2....D_N] for N cameras each D is a 5,
    "camera_frame_extrinsic_Rs" = [[camera_1_frame_1_R, camera_1_frame_2_R...], [camera_2_frame_1_R, camera_2_frame_2_R...]] and each R is a 3x3
    "camera_frame_extrinsic_Ts" = [[camera_1_frame_1_T, camera_1_frame_2_T...], [camera_2_frame_1_T, camera_2_frame_2_T...]] and each T is 3,
    "camera_frame_map_points_2d" = [[[camera_1_frame_1_points_2d],[camera_1_frame_2_points_2d],...],[[camera_2_frame_1_points_2d],[camera_1_frame_2_points_2d],...],...]
    the shape of this is of the form of a list of vector stacks which are the 2d points for each given camera frame. 
    
Actuall this is all we need to make into 1d (and that is what jax requires) and ide like to go more general


"""



from slampy.optimiser.lm import minimise
from slampy.utils.geometry import rvec_to_R, R_to_rvec, tvec_to_T, T_to_tvec, projection_error_2d_3d
from jax import numpy as jnp
import numpy as np

def loss_function(params_to_optimise, n_cameras, all_camera_frame_counts, all_camera_frame_point_counts, camera_frame_map_points_2d_3d_index):

    # Reshape the flattened params into the original structures
    [global_map_points_3d, converted_camera_intrinsic_Ks, converted_camera_intrinsic_Ds, converted_camera_frame_extrinsic_rvecs, converted_camera_frame_extrinsic_tvecs, converted_camera_frame_map_points_2d] = params_to_optimise
    
    #print("converted params unpacking", params_to_optimise)
    #print("converted params unpacking 3", params_to_optimise[3])
    #print("----------------")
    #print("converted params unpacking 4", params_to_optimise[4])
    #print("----------------")

    #print("global_map_points_3d", global_map_points_3d, global_map_points_3d.shape)
    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^24084238092438432890432890324890324890324890432890324089324890324890324^^")
    # map_points_3d and camera_extrinsics are being co-optimised
    total_error = 0
    for camera_idx in range(n_cameras):
        # Extract camera params
        
        flat_K = converted_camera_intrinsic_Ks[camera_idx]

        # Assign the optimized elements to their respective positions       
        K = jnp.array([
            flat_K[0],  # focal_length_x
            flat_K[1],  # focal_length_y
            flat_K[2],  # principal_point_x
            flat_K[3],  # principal_point_y
            flat_K[4],  # skew
            1,          # K[2, 2] fixed to 1
            0,          # K[1, 0] fixed to 0
            0,          # K[2, 0] fixed to 0
            0           # K[2, 1] fixed to 0
        ])

        # Reshape the new array into a 3x3 matrix
        K = jnp.reshape(K, (3, 3))

        #K = converted_camera_intrinsic_Ks[camera_idx]
        # we need 1x5 for D 9->13
        D = converted_camera_intrinsic_Ds[camera_idx]
        
        #print("camera_idx", camera_idx)
        #print("all_camera_frame_counts", all_camera_frame_counts)
        n_frames = all_camera_frame_counts[camera_idx]
        for frame_idx in range(n_frames):
            #print("converted_camera_frame_extrinsic_rvecs[camera_idx]", converted_camera_frame_extrinsic_rvecs[camera_idx])
            rvec = converted_camera_frame_extrinsic_rvecs[camera_idx][frame_idx]
            #print("rvec", rvec)
            #print("---------------------")
            tvec = converted_camera_frame_extrinsic_tvecs[camera_idx][frame_idx]
            map_points_2d = converted_camera_frame_map_points_2d[camera_idx][frame_idx]
            map_point_3d_indicies = camera_frame_map_points_2d_3d_index[camera_idx][frame_idx]
            
            
            
            map_points_3d = global_map_points_3d[map_point_3d_indicies]

            #print("map_points_2d", map_points_2d)
            #print("map_points_3d", map_points_3d)
            #print("map_point_3d_indicies", map_point_3d_indicies)
            
            #print("arrg", (K, D, rvec, tvec, map_points_2d, map_points_3d))
        
            
            # Calculate the projection error
            total_error += jnp.sum((projection_error_2d_3d(K, D, rvec, tvec, map_points_2d, map_points_3d))**2)
            #print("TTTOTAL ERROR", total_error)
    return (total_error)

"""


"params_to_optimise" set (needs to be flattened into a 1 dimensional buffer)

- "map_points_3d" a list of 3d points in world space. We have M of these so, we have a Mx3 matrix.
- "camera_intrinsic_params" for the N cameras, we have 5 (variable) K values and 5 D values, so in total we have (N x 10) of theses
- "camera_frame_extrinsic_params" this is more complicated as we have B_i frames (each a different number) for each camera
    and these frames have a 3 rvecs and 3 tvecs. So here we have 6 values per frame and B_i frames per camera and there are N cameras, 
    this adds some complexity as we need to pack/unpack this array. We will have sum(6 x B_i for i in range(N)) values in total, 
    it makes sense as this is an array and not a fixed size matrix for all cameras to organise this by the "fixed" per invocation number
    of cameras N and then to have B_i chunks of these the data objects (6) organised as blocks and rely on "params_static" "camera_frame_metadata"
    in order to know the size of each of these blocks per camera frame, for unpacking and repacking purposes.
- "camera_frame_map_points_2d" The 2d points per frame for each frame per camera for all cameras.
- "camera_frame_map_points_2d" we have, B_i frames for each ith camera, for each of those B_i camera frames we have L_i,j 2d points Lx2.
    A total of sum(2 x L_i,j x B_i  for i,k in B_i for i in range(N)). So this is a nested set of lists, from the highest level
    we can index by the camera index, then next level have another array for each of the frames of a given camera, and within that list
    is our 2d points.
- "camera_frame_map_points_2d_3d_index" same shape as camera_frame_map_points_2d however we have a single value for each camera frame
    2d point representing the index of where you could find the corresponding "map_point_3d" for this 2d camera frame feature,
    a total of sum(1 x L_i,j x B_i  for i,k in B_i for i in range(N))

"params_static" set (also needs to be flattened into a 1 dimensional buffer)

- "n_cameras" The number of cameras N
- "all_camera_frame_counts" The number of frames per N camera  (N x 1)
- "all_camera_frame_point_counts" The number of points that varies for each frame for each camera for all cameras.
- "camera_frame_map_points_2d_3d_index" same shape as camera_frame_map_points_2d however we have a single value for each camera frame
    2d point representing the index of where you could find the corresponding "map_point_3d" for this 2d camera frame feature,
    a total of sum(1 x L_i,j x B_i  for i,k in B_i for i in range(N))

"map_points_3d" = [[1,1,1],[1,1,1]] as a (M x 3)
"camera_intrinsic_params_Ks" = [K_1,K_2....K_N] for N cameras each K is a (3 x 3)
"camera_intrinsic_params_Ds" = [D_1,D_2....D_N] for N cameras each D is a 5,
"camera_frame_extrinsic_Rs" = [[camera_1_frame_1_R, camera_1_frame_2_R...], [camera_2_frame_1_R, camera_2_frame_2_R...]] and each R is a 3x3
"camera_frame_extrinsic_Ts" = [[camera_1_frame_1_T, camera_1_frame_2_T...], [camera_2_frame_1_T, camera_2_frame_2_T...]] and each T is 3,
"camera_frame_map_points_2d" = [[[camera_1_frame_1_points_2d],[camera_1_frame_2_points_2d],...],[[camera_2_frame_1_points_2d],[camera_1_frame_2_points_2d],...],...]
the shape of this is of the form of a list of vector stacks which are the 2d points for each given camera frame. 

    
"""

# camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index, camera_intrinsic, camera_frame_extrinsic, map_points_3d
def ba(map_points_3d, camera_intrinsic_Ks, camera_intrinsic_Ds, camera_frame_extrinsic_Rs, camera_frame_extrinsic_Ts, camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index):
    
    n_cameras = len(camera_intrinsic_Ks)
    all_camera_frame_counts = list(map(lambda x: len(x), [camera_R_poses for camera_R_poses in camera_frame_extrinsic_Rs]))
    all_camera_frame_point_counts=[]
    for camera in camera_frame_map_points_2d:
        all_camera_frame_point_counts=[]
        for frame in camera:
            frame_point_counts = []
            for points_mtx in frame:
                frame_point_counts.append(points_mtx.shape[0])
            all_camera_frame_point_counts.append(frame_point_counts)
    
    # Convert R's T's K's and D's to minimal format for optimisation
    
    converted_camera_frame_extrinsic_rvecs = [
        [R_to_rvec(np.asarray(R)) for R in Rs_per_frame_per_camera]
        for Rs_per_frame_per_camera in camera_frame_extrinsic_Rs
    ]

    converted_camera_frame_extrinsic_tvecs = [
        [T_to_tvec(np.asarray(T)) for T in Ts_per_frame_per_camera]
        for Ts_per_frame_per_camera in camera_frame_extrinsic_Ts
    ]
    
    converted_camera_intrinsic_Ks = [ np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]]) for K in camera_intrinsic_Ks ]       
    converted_camera_intrinsic_Ds = [ np.asarray(D) for D in camera_intrinsic_Ds ]       
        
    #converted_camera_frame_map_points_2d = [np.asarray(points) for cameras in camera_frame_map_points_2d for frames in cameras for points in frames]
    converted_camera_frame_map_points_2d = [
        [
            [np.asarray(point) for point in frame]
            for frame in cameras
        ]
        for cameras in camera_frame_map_points_2d
    ]
    
    #print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
    #print("map_points_3d", map_points_3d, map_points_3d.shape)
    
    params = [np.asarray(map_points_3d), converted_camera_intrinsic_Ks, converted_camera_intrinsic_Ds, converted_camera_frame_extrinsic_rvecs, converted_camera_frame_extrinsic_tvecs, converted_camera_frame_map_points_2d]
    args = (n_cameras, all_camera_frame_counts, all_camera_frame_point_counts, camera_frame_map_points_2d_3d_index)


    return minimise(loss_function, params, args=args)
    
    
    #print("minimise done", loss)

    # Retrieve optimised params
    #optimised_params = result.x
    #optimised_camera_intrinsic = optimised_params[:n_cameras * 8].reshape((n_cameras, 8))
    #optimised_camera_frame_extrinsics = optimised_params[n_cameras * 8:n_cameras * 8 + n_frames * 6].reshape((n_cameras, n_frames, 6))
    #optimised_map_points_3d = optimised_params[n_cameras * 8 + n_frames * 6:].reshape((-1, 3))
    #return optimised_camera_intrinsic, optimised_camera_frame_extrinsics, optimised_map_points_3d
    


def optimise():
    pass

