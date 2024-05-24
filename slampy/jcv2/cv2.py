from jax import numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize
from jax import jit, random
import jax
from slampy.optimiser.codec import serialise, deserialise

def undistort_points(K, D, map_points_2d):
    # Extract distortion coefficients
    k1, k2, p1, p2, k3 = D

    # Unpack the intrinsic matrix
    fx, fy, cx, cy, s = K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]

    # Normalize coordinates
    normalized_points = ((map_points_2d - jnp.array([[cx], [cy]])) / jnp.array([[fx], [fy]]))

    # Apply radial distortion correction
    r2 = normalized_points[0, :]**2 + normalized_points[1, :]**2
    radial_correction = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    # Apply tangential distortion correction
    p_correction_x = 2 * p1 * normalized_points[0, :] * normalized_points[1, :] + p2 * (r2 + 2 * normalized_points[0, :]**2)
    p_correction_y = p1 * (r2 + 2 * normalized_points[1, :]**2) + 2 * p2 * normalized_points[0, :] * normalized_points[1, :]

    # Apply skew factor
    skew_correction = s * normalized_points[0, :]

    # Apply distortion correction
    undistorted_normalized_points = jnp.vstack([
        normalized_points[0, :] * radial_correction + p_correction_x + skew_correction,
        normalized_points[1, :] * radial_correction + p_correction_y
    ])

    # Rescale to pixel coordinates
    undistorted_points = jnp.vstack([
        undistorted_normalized_points[0, :] * fx + cx,
        undistorted_normalized_points[1, :] * fy + cy
    ])

    return undistorted_points

def rvec_to_R(rvec):
    #print("got back an rvec", rvec)
    theta = jnp.linalg.norm(rvec)

    axis = rvec / theta
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])

    R_nonzero = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * jnp.dot(K, K)

    # Special case handling for theta == 0
    R_zero = jnp.eye(3)

    # Use jnp.where to conditionally select the result
    R = jnp.where(theta == 0, R_zero, R_nonzero)

    return R

def R_to_rvec(R):
    if not isinstance(R, np.ndarray):
        raise ValueError("Input 'R' must be a NumPy array.")

    if R.shape != (3, 3):
        raise ValueError("Input 'R' must be a 3x3 matrix.")

    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2.0)

    if theta == 0:
        return np.zeros(3)

    axis = 1 / (2 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    # Ensure the axis is a unit vector
    axis /= np.linalg.norm(axis)

    # Scale the axis by the rotation angle to obtain the rotation vector
    rvec = theta * axis
    return rvec

def project_3d_points(K, D, rvec, tvec, points_3d):
    R = rvec_to_R(rvec)
    T = tvec_to_T(tvec)
    P = jnp.dot(K,jnp.hstack([R, T]))
    points_4d = jnp.concatenate((points_3d, jnp.ones((len(points_3d), 1))), axis=1)
    points_2d = jnp.dot(P, points_4d.T).T[2,:]
    norm_points_2d = undistort_points(K, D, points_2d)
    return norm_points_2d
    


def projection_error_2d_3d(K, D, rvec, tvec, map_points_2d, map_points_3d):
    
    R = rvec_to_R(rvec) # 3x3
    # tvec is 3,
    T = tvec_to_T(tvec) #3x1
    
    # hstack jnp.hstack(R, T) is putting a 3x3 and a 3x1 so we have one more column so 3x4
    
    # K is 3x3
    
    # 3x3 dot 3x4 is a 3x4
    
    # transformation matrix

    P = jnp.dot(K,jnp.hstack([R, T])) # 3x4

    # get 3d homogeneous points
    # map points 3d is Nx3 and map points give one extra column so Nx4
    map_points_4d = jnp.concatenate((map_points_3d, jnp.ones((len(map_points_3d), 1))), axis=1)
    
    
    # get 2d distorted map points
    # 3x4 dot Nx4 does not work so need to transpose map_points_4d to 4xN
    # 3x4 dot 4xN is 3xN
    # so 3x4 dot 4xN is 3xN
    # it would be nice is we had our data in rows for the x,y,z,w
    # so do transpose of jnp.dot(P, map_points_4d.T) is Nx3
    # then we want to select the first two rows x,y and all of the columns so [:2,:]

    map_points_2d = jnp.dot(P, map_points_4d.T).T[2,:]
    
    
    norm_map_points_2d = undistort_points(K, D, map_points_2d)
    
    # next calculate the error
    # find the distance between two sets of 2d points
    
    error_2d_points = norm_map_points_2d - map_points_2d
    
    
    return error_2d_points

    #total_project_error = jnp.sum(jnp.linalg.norm(error_2d_points, axis=2))


def tvec_to_T(tvec):
    #T = jnp.eye(4)
    #T[:3, 3] = tvec
    return tvec.reshape((3, 1))

def T_to_tvec(T):
    return T.flatten()

################################################# Solve PNP RANSAC
def pnp_objective(params, pts_3d, pts_2d, K, D):
        rvec, tvec = params[:3], params[3:]
        proj_pts_2d = project_3d_points(pts_3d, rvec, tvec, K, D) # project_points(pts_3d, rvec, tvec, K, D)
        reprojection_error = jnp.sum((proj_pts_2d - pts_2d) ** 2)
        return reprojection_error

def solvePnPRansac(pts_3d, pts_2d, K, D, iterations=3000, confidence=0.99999):
    key = random.PRNGKey(0)  # Use a fixed key for reproducibility, you might want to change this
    sampled_indices = random.choice(key, pts_3d.shape[0], shape=(iterations,), replace=True)

    # RANSAC loop
    best_params = None
    best_inliers = jnp.zeros(pts_3d.shape[0], dtype=bool)
   
    for _ in range(iterations):
        sample_indices = sampled_indices[_]
        sampled_pts_3d = pts_3d[sample_indices]
        sampled_pts_2d = pts_2d[sample_indices]

        # Use RANSAC to estimate parameters
        result = minimize(pnp_objective, jnp.zeros(6), args=(sampled_pts_3d, sampled_pts_2d, K, D), method='trust-constr', jac='2-point')
        refined_params = result.x

        # Evaluate the refined solution
        rvec, tvec = refined_params[:3], refined_params[3:]
        proj_pts_2d = project_3d_points(pts_3d, rvec, tvec, K, D)
        reprojection_error = jnp.sum((proj_pts_2d - pts_2d) ** 2)
        inliers_mask = reprojection_error < confidence

        # Update best solution if current is better
        if jnp.sum(inliers_mask) > jnp.sum(best_inliers):
            best_params = refined_params
            best_inliers = inliers_mask

    return best_params, best_inliers

# REFINE PNP LM

def refinePnPLM(pts_3d, pts_2d, K, D, initial_rvec, initial_tvec):
    # Initialize LM parameters
    initial_params = jnp.concatenate([initial_rvec, initial_tvec])

    # Use LM to refine parameters
    result = minimize(pnp_objective, initial_params, args=(pts_3d, pts_2d, K, D), method='trust-constr', jac='2-point')
    refined_params = result.x

    return refined_params

##### FUNDAMENTAL AND ESSENTIAL


def fundamental_matrix_model(params):
    return jnp.array(params).reshape((3, 3))

def compute_residuals(params, points1, points2):
    F = fundamental_matrix_model(params)
    residuals = jnp.sum(points2 * (F @ points1[:, None]), axis=-1)
    return residuals

def loss_method(params, *args):
    points1, points2, inlier_threshold = args
    residuals = compute_residuals(params, points1, points2)
    inliers = jnp.abs(residuals) < inlier_threshold
    total_loss = jnp.sum(inliers)
    return total_loss, inliers

def minimise_lm(udf_error_function, initial_params_estimate, args=(), epoch=10, damping_factor=0.1):
    loss_func = lambda x: udf_error_function(x, *args)[0]
    params = initial_params_estimate
    
    for _ in range(epoch):
        flat_params, flat_params_meta = serialise(params)

        loss, udf_additional_return_values = udf_error_function(params, *args)
        wrapped_udf_flat = lambda x: udf_error_function(deserialise(x, flat_params_meta), *args)[0]
        wrapped_udf_grad_wrt_params = jax.grad(wrapped_udf_flat)
        udf_jac = wrapped_udf_grad_wrt_params(flat_params)

        wrapped_udf_hessian_wrt_params = jax.hessian(wrapped_udf_flat)
        udf_hessian = wrapped_udf_hessian_wrt_params(flat_params)

        damping_matrix = jnp.eye(len(flat_params)) * damping_factor

        change_in_params = jnp.linalg.solve(udf_hessian + damping_matrix, -udf_jac.flatten())
        params += change_in_params.reshape(params.shape)

    return params, *udf_additional_return_values

def ransac_iteration(key, points1, points2, inlier_threshold, method):
    sample_indices = random.choice(key, points1.shape[0], shape=(8,), replace=False)
    sampled_points1 = points1[sample_indices]
    sampled_points2 = points2[sample_indices]

    initial_params = jnp.zeros(9)

    if method == "FM_RANSAC":
        refined_params, _ = ransac_minimise(sampled_points1, sampled_points2, inlier_threshold=inlier_threshold, epoch=10)
    else:
        refined_params, _ = minimise_lm(loss_method, initial_params, args=(sampled_points1, sampled_points2, inlier_threshold), epoch=10)

    residuals = compute_residuals(refined_params, points1, points2)
    inliers = jnp.abs(residuals) < inlier_threshold

    return refined_params, inliers

def ransac_minimise(points1, points2, inlier_threshold=1.0, epoch=10, method="FM_RANSAC"):
    best_params = None
    best_inliers = jnp.zeros(points1.shape[0], dtype=bool)

    for _ in range(epoch):
        key = random.PRNGKey(random.randint(0, 2**32 - 1))
        params, inliers = ransac_iteration(key, points1, points2, inlier_threshold, method)

        if jnp.sum(inliers) > jnp.sum(best_inliers):
            best_inliers = inliers
            best_params = params

    return best_params, best_inliers

def ransac_fundamental_matrix(points1, points2, inlier_threshold=1.0, epoch=10, min_points=8):
    best_F_matrices = []
    best_inliers = jnp.zeros(points1.shape[0], dtype=bool)

    for _ in range(epoch):
        key = random.PRNGKey(random.randint(0, 2**32 - 1))
        sampled_indices = random.choice(key, points1.shape[0], shape=(min_points,), replace=False)
        sampled_points1 = points1[sampled_indices]
        sampled_points2 = points2[sampled_indices]

        params, inliers = ransac_iteration(key, sampled_points1, sampled_points2, inlier_threshold, method="FM_RANSAC")

        if jnp.sum(inliers) >= min_points and jnp.sum(inliers) > jnp.sum(best_inliers):
            best_inliers = inliers
            best_F = fundamental_matrix_model(params)
            best_F_matrices = jnp.concatenate([best_F_matrices, jnp.expand_dims(best_F, axis=-1)], axis=-1)

    return best_F_matrices

def recover_E_from_F(F_matrix, intrinsics1, intrinsics2):
    E_matrix = intrinsics2.T @ F_matrix @ intrinsics1
    return E_matrix

def compute_essential_matrices_from_fundamentals(fundamental_matrices):
    intrinsics1 = jnp.eye(3)
    intrinsics2 = jnp.eye(3)
    essential_matrices = []

    for F_matrix in jnp.split(fundamental_matrices, len(fundamental_matrices) // 3, axis=-1):
        E_matrix = recover_E_from_F(F_matrix)
        essential_matrices = jnp.concatenate([essential_matrices, jnp.expand_dims(E_matrix, axis=-1)], axis=-1)

    return essential_matrices

def skew_matrix(v):
    """Skew-symmetric matrix from a 3D vector."""
    return jnp.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

def decomposeEssentialMatrix(E):
    """Decompose essential matrix into rotation and translation."""
    U, S, V = jnp.linalg.svd(E)
    W = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ V.T
    R2 = U @ W.T @ V.T
    T = U[:, 2]

    return R1, R2, T

@jit
def recoverPose(frame_1_pts_2d_epipolar, frame_2_pts_2d_epipolar, K1, D1, K2, D2, E, prob=0.99999, threshold=0.6):
    # Decompose essential matrix
    R1, R2, T = decomposeEssentialMatrix(E)

    # Linear triangulation to obtain 3D points
    P1 = K1 @ jnp.hstack([jnp.eye(3), jnp.zeros((3, 1))])
    

"""
# Example usage:
# Assuming points1 (query) and points2 (train) are corresponding points in two images
points1 = jnp.array([[x1, y1] for x1, y1 in zip(range(10), range(10))])
points2 = jnp.array([[x2, y2] for x2, y2 in zip(range(10, 20), range(10, 20))])

# RANSAC to find the best parameters and inliers using FM_RANSAC method
best_F_matrices = ransac_fundamental_matrix(points1, points2, inlier_threshold=1.0)

# Unpack the F matrices using np.split
for _F in jnp.split(best_F_matrices, len(best_F_matrices) // 3, axis=-1):
    print("F Matrix:", _F)

# Compute essential matrices from fundamental matrices
essential_matrices = compute_essential_matrices_from_fundamentals(best_F_matrices)

# Recover poses from essential matrices
best_n_inliers = 0
ret = None
for _E in jnp.split(essential_matrices, len(essential_matrices) // 3, axis=-1):
    n_pose_inliers, E, R2, T2, pose_mask = recover_pose_result = cv2.recoverPose(frame_1_pts_2d_epipolar, frame_2_pts_2d_epipolar, K1, D1, K2, D2, _E, prob=0.99999, threshold=0.6)
    if n_pose_inliers > best_n_inliers:
        best_n_inliers = n_pose_inliers
        ret = recover_pose_result

# Use 'ret' for further processing
print("Best Poses:", ret)
"""