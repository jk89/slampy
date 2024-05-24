from scipy import minimize
from numba import njit, jit, prange
import numpy as np

"""
Bundle Adjustment (BA) Pipeline using Numba Acceleration

This code implements a Bundle Adjustment (BA) pipeline for refining camera intrinsic and extrinsic params as well as 3D map points. The Numba library is utilized to accelerate critical functions, ensuring efficient computation.

Key Components:
- Projection Model Functions: Functions like undistort_points, rvec_to_R, and projection_error handle the transformation from 3D points to 2D points, considering camera intrinsic params, distortion coefficients, rotation vector, translation vector, and the projection matrix.

- Loss Function: The loss_function calculates the total reprojection error across all cameras, frames, and corresponding 2D-3D point pairs. It takes a flattened array of params representing camera intrinsic/extrinsic params and 3D points.

- Bundle Adjustment: The ba function uses the minimize function from SciPy to optimise params by minimizing the reprojection error. Initial params are flattened and concatenated for optimization, and the resulting optimised params are reshaped back into their respective structures.

"""

@jit(cache=True, nogil=True)
def undistort_points(K, D, map_points_2d):
    # Extract distortion coefficients
    k1, k2, p1, p2, k3 = D

    # Unpack the intrinsic matrix
    fx, fy, cx, cy, s = K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]

    # Normalize coordinates
    normalized_points = ((map_points_2d - np.array([[cx], [cy]])) / np.array([[fx], [fy]]))

    # Apply radial distortion correction
    r2 = normalized_points[0, :]**2 + normalized_points[1, :]**2
    radial_correction = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    # Apply tangential distortion correction
    p_correction_x = 2 * p1 * normalized_points[0, :] * normalized_points[1, :] + p2 * (r2 + 2 * normalized_points[0, :]**2)
    p_correction_y = p1 * (r2 + 2 * normalized_points[1, :]**2) + 2 * p2 * normalized_points[0, :] * normalized_points[1, :]

    # Apply skew factor
    skew_correction = s * normalized_points[0, :]

    # Apply distortion correction
    undistorted_normalized_points = np.vstack([
        normalized_points[0, :] * radial_correction + p_correction_x + skew_correction,
        normalized_points[1, :] * radial_correction + p_correction_y
    ])

    # Rescale to pixel coordinates
    undistorted_points = np.vstack([
        undistorted_normalized_points[0, :] * fx + cx,
        undistorted_normalized_points[1, :] * fy + cy
    ])

    return undistorted_points

@jit(cache=True, nogil=True)
def rvec_to_R(rvec):
    theta = np.linalg.norm(rvec)
    if theta == 0:
        return np.eye(3)

    axis = rvec / theta
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R

@jit(cache=True, nogil=True)
def projection_error(K, D, rvec, tvec, map_points_2d, map_points_3d):
    
    R = rvec_to_R(rvec) # 3x3
    # tvec is 3,
    T = tvec.reshape(1,3) #3x1
    
    # hstack np.hstack(R, T) is putting a 3x3 and a 3x1 so we have one more column so 3x4
    
    # K is 3x3
    
    # 3x3 dot 3x4 is a 3x4
    
    # transformation matrix
    P = np.dot(K,np.hstack([R, T])) # 3x4
    
    # get 3d homogeneous points
    # map points 3d is Nx3 and map points give one extra column so Nx4
    map_points_4d = np.concatenate((map_points_3d, np.ones((len(map_points_3d), 1))), axis=1)
    
    # get 2d distorted map points
    # 3x4 dot Nx4 does not work so need to transpose map_points_4d to 4xN
    # 3x4 dot 4xN is 3xN
    # so 3x4 dot 4xN is 3xN
    # it would be nice is we had our data in rows for the x,y,z,w
    # so do transpose of np.dot(P, map_points_4d.T) is Nx3
    # then we want to select the first two rows x,y and all of the columns so [:2,:]
    map_points_2d = np.dot(P, map_points_4d.T).T[2,:]
    
    norm_map_points_2d = undistort_points(K, D, map_points_2d)
    
    # next calculate the error
    # find the distance between two sets of 2d points
    
    error_2d_points = norm_map_points_2d - map_points_2d
    
    total_project_error = np.sum(np.linalg.norm(error_2d_points, axis=2))
    return total_project_error

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

"params_static" set (also needs to be flattened into a 1 dimensional buffer)

- "n_cameras" The number of cameras N
- "all_camera_frame_counts" The number of frames per N camera  (N x 1)
- "all_camera_frame_point_counts" The number of points that varies for each frame for each camera for all cameras.
- "camera_frame_map_points_2d" The 2d points per frame for each frame per camera for all cameras.
- "camera_frame_map_points_2d" we have, B_i frames for each ith camera, for each of those B_i camera frames we have L_i,j 2d points Lx2.
    A total of sum(2 x L_i,j x B_i  for i,k in B_i for i in range(N)). So this is a nested set of lists, from the highest level
    we can index by the camera index, then next level have another array for each of the frames of a given camera, and within that list
    is our 2d points.
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






@njit(parallel=True, cache=True, nogil=True)
def loss_function(params_to_optimise, camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index):
    n_cameras = len(camera_frame_map_points_2d)
    n_frames = len(camera_frame_map_points_2d[0])

    # Reshape the flattened params into the original structures
    camera_intrinsic_params = params_to_optimise[:n_cameras * 8].reshape((n_cameras, 8))
    camera_frame_extrinsic_params = params_to_optimise[n_cameras * 8:n_cameras * 8 + n_frames * 6].reshape((n_cameras, n_frames, 6))
    map_points_3d_params = params_to_optimise[n_cameras * 8 + n_frames * 6:].reshape((-1, 3))
    
    # map_points_3d and camera_extrinsics are being co-optimised
    total_error = 0
    n_cameras = camera_intrinsic_params.shape[0]
    for camera_idx in prange(n_cameras):
        # Extract camera params
        # we need a 3x3 for K 0->8
        K = camera_intrinsic_params[camera_idx,:8].reshape(3,3)
        # we need 1x5 for D 9->13
        D = camera_intrinsic_params[camera_idx:9:13]
        
        n_frames = camera_frame_extrinsic_params[camera_idx].shape[0]
        for frame_idx in prange(n_frames):
            # we need 1x3 for rvec 0-2
            rvec = camera_frame_extrinsic_params[camera_idx][:2]
            # we need 1x3 for tvec 3->5
            tvec = camera_frame_extrinsic_params[camera_idx][3:]
            map_points_2d = camera_frame_map_points_2d[camera_idx][frame_idx]
            map_point_3d_indicies = camera_frame_map_points_2d_3d_index[camera_idx][frame_idx]
            map_points_3d = map_points_3d_params[map_point_3d_indicies]
        
            
            # Calculate the projection error
            total_error += projection_error(K, D, rvec, tvec, map_points_2d, map_points_3d)
    return total_error

@jit(cache=True, nogil=True)
def ba(camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index, camera_intrinsic, camera_frame_extrinsic, map_points_3d):
    initial_params = np.concatenate([camera_intrinsic.flatten(),
                                  camera_frame_extrinsic.flatten(),
                                  map_points_3d.flatten()])
    result = minimize(loss_function, initial_params,
                    args=(camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index),
                    method='lm')

    n_cameras = camera_intrinsic.shape[0]
    n_frames = camera_frame_extrinsic.shape[0]
    
    # Retrieve optimised params
    optimised_params = result.x
    optimised_camera_intrinsic = optimised_params[:n_cameras * 8].reshape((n_cameras, 8))
    optimised_camera_frame_extrinsics = optimised_params[n_cameras * 8:n_cameras * 8 + n_frames * 6].reshape((n_cameras, n_frames, 6))
    optimised_map_points_3d = optimised_params[n_cameras * 8 + n_frames * 6:].reshape((-1, 3))
    return optimised_camera_intrinsic, optimised_camera_frame_extrinsics, optimised_map_points_3d
    
