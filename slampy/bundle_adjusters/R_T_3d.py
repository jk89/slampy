import numpy as np
from slampy.optimiser.lm import minimise
from slampy.utils.geometry import rvec_to_R, R_to_rvec, tvec_to_T, T_to_tvec, projection_error_2d_3d
from jax import numpy as jnp

"""
Bundle adjustment optimizing 3D points and extrinsics only.
Intrinsic K, D and 2D measurements fixed in args.
"""

def loss_function(params_to_optimise,
                  n_cameras,
                  all_camera_frame_counts,
                  all_camera_frame_point_counts,
                  camera_intrinsic_params_Ks_flat,
                  camera_intrinsic_params_Ds,
                  camera_frame_map_points_2d,
                  camera_frame_map_points_2d_3d_index):
    # Unpack optimized params: map_points_3d, rvecs, tvecs
    map_points_3d, converted_rvecs, converted_tvecs = params_to_optimise
    total_error = 0.0

    for camera_idx in range(n_cameras):
        flat_K = camera_intrinsic_params_Ks_flat[camera_idx]
        # Assign the optimized elements to their respective positions       
        """K = jnp.array([
            flat_K[0],  # focal_length_x
            flat_K[1],  # focal_length_y
            flat_K[2],  # principal_point_x
            flat_K[3],  # principal_point_y
            flat_K[4],  # skew
            1.0,        # K[2, 2] fixed to 1
            0.0,        # K[1, 0] fixed to 0
            0.0,        # K[2, 0] fixed to 0
            0.0         # K[2, 1] fixed to 0
        ])
        # Reshape to 3x3
        K = K.reshape((3, 3))"""

        K = jnp.array([
            flat_K[0], flat_K[4], flat_K[2],
            0.0,        flat_K[1], flat_K[3],
            0.0,        0.0,        1.0
        ]).reshape((3, 3))
        
        D = camera_intrinsic_params_Ds[camera_idx]

        n_frames = all_camera_frame_counts[camera_idx]
        for frame_idx in range(n_frames):
            rvec = converted_rvecs[camera_idx][frame_idx]
            tvec = converted_tvecs[camera_idx][frame_idx]
            pts2d = camera_frame_map_points_2d[camera_idx][frame_idx]
            idx3d = camera_frame_map_points_2d_3d_index[camera_idx][frame_idx]
            pts3d = map_points_3d[idx3d]

            err = projection_error_2d_3d(K, D, rvec, tvec, pts2d, pts3d)
            #print(f"Camera {camera_idx}, Frame {frame_idx}, err:", err)
            total_error += jnp.sum(err**2)

    return total_error


def ba(map_points_3d,
       camera_intrinsic_params_Ks,
       camera_intrinsic_params_Ds,
       camera_frame_extrinsic_Rs,
       camera_frame_extrinsic_Ts,
       camera_frame_map_points_2d,
       camera_frame_map_points_2d_3d_index):
    # counts for unpacking
    n_cameras = len(camera_intrinsic_params_Ks)
    all_camera_frame_counts = [len(frames) for frames in camera_frame_extrinsic_Rs]
    all_camera_frame_point_counts = [[pts.shape[0] for pts in frames]
                                     for frames in camera_frame_map_points_2d]

    # convert extrinsics
    converted_rvecs = [[R_to_rvec(R) for R in frames]
                       for frames in camera_frame_extrinsic_Rs]
    converted_tvecs = [[T_to_tvec(T) for T in frames]
                       for frames in camera_frame_extrinsic_Ts]

    # Add perturbation to zero rotation vectors # FIXME
    eps = 1e-8
    for cam_frames in converted_rvecs:
        for i, rvec in enumerate(cam_frames):
            if np.linalg.norm(rvec) < eps:
                cam_frames[i] = np.array([eps, 0, 0])  # tiny rotation around x-axis


    # minimal intrinsics
    camera_intrinsic_params_Ks_flat = [
        np.array([K[0,0], K[1,1], K[0,2], K[1,2], K[0,1]])
        for K in camera_intrinsic_params_Ks
    ]

    # pack params_to_optimise
    params = [np.asarray(map_points_3d), converted_rvecs, converted_tvecs]

    # static args
    args = (
        n_cameras,
        all_camera_frame_counts,
        all_camera_frame_point_counts,
        camera_intrinsic_params_Ks_flat,
        camera_intrinsic_params_Ds,
        camera_frame_map_points_2d,
        camera_frame_map_points_2d_3d_index,
    )

    return minimise(loss_function, params, args=args)

