from jax import numpy as jnp
import numpy as np

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