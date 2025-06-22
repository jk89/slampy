import numpy as np
import unittest
from slampy.bundle_adjusters.R_T_3d import ba
from slampy.optimiser.codec import serialise, deserialise
from jax import numpy as jnp

class TestCubeSceneBundleAdjustment(unittest.TestCase):

    def generate_cube_scene(self, num_cameras=3, max_frames_per_camera=2):
        # Define cube corners (8 points)
        cube_points_3d = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=float)

        # Intrinsics: simple pinhole camera model for each camera
        K = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=float)
        camera_intrinsic_params_Ks = [K.copy() for _ in range(num_cameras)]
        camera_intrinsic_params_Ds = [np.zeros(5) for _ in range(num_cameras)]

        # Camera extrinsics: fixed rotations and translations per frame
        camera_frame_extrinsic_Rs = []
        camera_frame_extrinsic_Ts = []
        camera_frame_map_points_2d = []
        camera_frame_map_points_2d_3d_index = []

        for cam_idx in range(num_cameras):
            Rs = []
            Ts = []
            frames_2d = []
            frames_idx = []
            for frame_idx in range(max_frames_per_camera):
                # Simple rotation around Y axis by different angles per camera and frame
                angle = np.deg2rad(15 * cam_idx + 10 * frame_idx)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                R = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ], dtype=float)
                t = np.array([cam_idx*2 + frame_idx*0.5, 0, 5], dtype=float)  # shifted along X, fixed Z=5

                Rs.append(R)
                Ts.append(t)

                # Project 3D cube points to 2D image points
                points_cam = (R @ cube_points_3d.T).T + t
                points_proj = points_cam[:, :2] / points_cam[:, 2, np.newaxis]  # perspective divide
                points_proj = (points_proj * np.array([K[0, 0], K[1, 1]])) + np.array([K[0, 2], K[1, 2]])
                
                frames_2d.append(points_proj)
                frames_idx.append(np.arange(len(cube_points_3d)))

            camera_frame_extrinsic_Rs.append(Rs)
            camera_frame_extrinsic_Ts.append(Ts)
            camera_frame_map_points_2d.append(frames_2d)
            camera_frame_map_points_2d_3d_index.append(frames_idx)

        return (cube_points_3d,
                camera_intrinsic_params_Ks,
                camera_intrinsic_params_Ds,
                camera_frame_extrinsic_Rs,
                camera_frame_extrinsic_Ts,
                camera_frame_map_points_2d,
                camera_frame_map_points_2d_3d_index)

    def test_ba_with_cube_scene(self):
        data = self.generate_cube_scene(num_cameras=3, max_frames_per_camera=2)


        (map_points_3d, camera_intrinsic_params_Ks, camera_intrinsic_params_Ds,
         camera_frame_extrinsic_Rs, camera_frame_extrinsic_Ts,
         camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index) = data
        
        print("map_points_3d shape:", map_points_3d.shape)
        print("camera_intrinsic_params_Ks shapes:", [[K.shape if isinstance(K, np.ndarray) else K for K in camera_intrinsic_params_Ks]])
        print("camera_intrinsic_params_Ds shapes:", [[D.shape if isinstance(D, np.ndarray) else D for D in camera_intrinsic_params_Ds]])
        print("camera_frame_extrinsic_Rs shapes:", [[[R.shape if isinstance(R, np.ndarray) else R for R in Rs] for Rs in camera_frame_extrinsic_Rs]])
        print("camera_frame_extrinsic_Ts shapes:", [[[T.shape if isinstance(T, np.ndarray) else T for T in Ts] for Ts in camera_frame_extrinsic_Ts]])
        print("camera_frame_map_points_2d shapes:", [[[p.shape if isinstance(p, np.ndarray) else p for p in frame] for frame in camera_frame_map_points_2d]])
        print("camera_frame_map_points_2d_3d_index shapes:", [[idx if isinstance(idx, list) else [idx] for idx in frame] for frame in camera_frame_map_points_2d_3d_index])
        
        print("map_points_3d:", map_points_3d)
        print("camera_intrinsic_params_Ks:", camera_intrinsic_params_Ks)
        print("camera_intrinsic_params_Ds:", camera_intrinsic_params_Ds)
        print("camera_frame_extrinsic_Rs:", camera_frame_extrinsic_Rs)
        print("camera_frame_extrinsic_Ts:", camera_frame_extrinsic_Ts)
        print("camera_frame_map_points_2d:", camera_frame_map_points_2d)
        print("camera_frame_map_points_2d_3d_index:", camera_frame_map_points_2d_3d_index)


        original_map_points_3d = data[0]
        results = ba(*data)

        print("results", results)
        print("original_map_points_3d", original_map_points_3d)

        optimized_points_3d = results[0][0]

        optimized_points_3d = np.asarray(optimized_points_3d)
        original_map_points_3d = np.asarray(original_map_points_3d)

        # Check shape matches
        self.assertEqual(optimized_points_3d.shape, original_map_points_3d.shape)

        # Check the optimized points are close to original points within epsilon
        epsilon = 1e-2  # Adjust threshold as needed
        max_diff = np.max(np.abs(np.array(optimized_points_3d) - original_map_points_3d))
        self.assertLessEqual(max_diff, epsilon, f"Max difference {max_diff} exceeds tolerance {epsilon}")

        print("Max difference between optimized and original points:", max_diff)


if __name__ == '__main__':
    unittest.main()