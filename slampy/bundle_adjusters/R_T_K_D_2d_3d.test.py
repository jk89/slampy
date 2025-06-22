import numpy as np
import unittest
from slampy.bundle_adjusters.R_T_K_D_2d_3d import ba
from slampy.optimiser.codec import serialise, deserialise
from jax import numpy as jnp

def recursive_array_equal(arr1, arr2):
    """
    Recursively compare JAX arrays. If both inputs are not JAX arrays, directly compare them.
    If one is a JAX array and the other is not, cast the non-JAX array to JAX array and then compare.
    If both inputs are JAX arrays, compare their numpy views.
    """
    if isinstance(arr1, jnp.ndarray) and isinstance(arr2, jnp.ndarray):
        return np.array_equal(arr1, arr2)

    if isinstance(arr1, jnp.ndarray):
        arr2 = jnp.asarray(arr2)
    elif isinstance(arr2, jnp.ndarray):
        arr1 = jnp.asarray(arr1)

    if isinstance(arr1, list) and isinstance(arr2, list):
        return all(recursive_array_equal(subarr1, subarr2) for subarr1, subarr2 in zip(arr1, arr2))

    return jnp.array_equal(arr1, arr2)

def generate_random_cameras(num_cameras, max_frames_per_camera=5, max_camera_frame_points = 3, num_map_points=10):
    # Generate random 3D map points
    map_points_3d = np.random.rand(num_map_points, 3)

    # Generate random camera intrinsic parameters (Ks and Ds)
    camera_intrinsic_params_Ks = [np.random.rand(3, 3) for _ in range(num_cameras)]
    camera_intrinsic_params_Ds = [np.random.rand(5) for _ in range(num_cameras)]
    
    # for each camera
    
    camera_frame_extrinsic_Rs = []
    camera_frame_extrinsic_Ts = []
    camera_frame_map_points_2d = []
    camera_frame_map_points_2d_3d_index = []
    for camera_idx in range(num_cameras):
        # Create a random number of frames
        camera_frames_per_camera = np.random.randint(max_frames_per_camera) + 1
        
        frame_Rs = []
        frame_Ts = []
        frame_2d = []
        frame_2d_3d_index = []
        
        print("camera_frames_per_camera", camera_frames_per_camera, max_frames_per_camera)
        
        for frame_idx in range(camera_frames_per_camera):
            frame_Rs.append(np.random.rand(3, 3))
            frame_Ts.append(np.random.rand(3, 1))
            camera_frame_points = np.random.randint(max_camera_frame_points) + 1
            print("camera_frame_points", camera_frame_points, max_camera_frame_points)
            camera_frame_2d_points = np.random.rand(camera_frame_points, 2)
            frame_2d_3d_indicies = np.random.choice(num_map_points, camera_frame_points, replace=False)
            print("frame_2d_3d_indicies", frame_2d_3d_indicies)
            frame_2d.append(camera_frame_2d_points)
            frame_2d_3d_index.append(frame_2d_3d_indicies)
            
        camera_frame_extrinsic_Rs.append(frame_Rs)
        camera_frame_extrinsic_Ts.append(frame_Ts)
        camera_frame_map_points_2d.append(frame_2d)
        camera_frame_map_points_2d_3d_index.append(frame_2d_3d_index)
        
    return map_points_3d, camera_intrinsic_params_Ks, camera_intrinsic_params_Ds, \
           camera_frame_extrinsic_Rs, camera_frame_extrinsic_Ts, \
           camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index


class TestRandomParameters(unittest.TestCase):    
    def test_random_parameters(self):
        ret = generate_random_cameras(3,2)
        [map_points_3d, camera_intrinsic_params_Ks, camera_intrinsic_params_Ds, camera_frame_extrinsic_Rs, camera_frame_extrinsic_Ts, camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index] = ret
        print("map_points_3d shape:", map_points_3d.shape)
        print("camera_intrinsic_params_Ks shapes:", [[K.shape if isinstance(K, np.ndarray) else K for K in camera_intrinsic_params_Ks]])
        print("camera_intrinsic_params_Ds shapes:", [[D.shape if isinstance(D, np.ndarray) else D for D in camera_intrinsic_params_Ds]])
        print("camera_frame_extrinsic_Rs shapes:", [[[R.shape if isinstance(R, np.ndarray) else R for R in Rs] for Rs in camera_frame_extrinsic_Rs]])
        print("camera_frame_extrinsic_Ts shapes:", [[[T.shape if isinstance(T, np.ndarray) else T for T in Ts] for Ts in camera_frame_extrinsic_Ts]])
        print("camera_frame_map_points_2d shapes:", [[[p.shape if isinstance(p, np.ndarray) else p for p in frame] for frame in camera_frame_map_points_2d]])
        print("camera_frame_map_points_2d_3d_index shapes:", [[idx if isinstance(idx, list) else [idx] for idx in frame] for frame in camera_frame_map_points_2d_3d_index])
        self.assertTrue(True), f"Random camera params generated"

    def test_serialise_deserialise_with_mock(self):
        ret = generate_random_cameras(3,2)
        complex_example = list(ret)
        complex_example_copy = complex_example.copy()
        flat_data_1, metadata = serialise(complex_example)
        flat_data_2, metadata = serialise(complex_example)
        
        data_1 = deserialise(flat_data_1, metadata)
        data_2 = deserialise(flat_data_2, metadata)
        
        self.assertTrue(recursive_array_equal(complex_example, data_1), f"Failed for by ref check 1")
        self.assertTrue(recursive_array_equal(complex_example_copy, data_1), f"Failed for by ref check 1")
        self.assertTrue(recursive_array_equal(complex_example, data_2), f"Failed for by ref check 3")
        self.assertTrue(recursive_array_equal(complex_example_copy, data_2), f"Failed for by ref check 4")
        
        
        print("data_1[3]", data_1[3])
        print("data_1[4]", data_1[4])
        print("data_2[3]", data_2[3])
        print("data_2[4]", data_2[4])
       
        
    def test_ba(self):
        ret = generate_random_cameras(3,2)
        [map_points_3d, camera_intrinsic_params_Ks, camera_intrinsic_params_Ds, camera_frame_extrinsic_Rs, camera_frame_extrinsic_Ts, camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index] = ret
        print("map_points_3d shape:", map_points_3d.shape)
        print("camera_intrinsic_params_Ks shapes:", [[K.shape if isinstance(K, np.ndarray) else K for K in camera_intrinsic_params_Ks]])
        print("camera_intrinsic_params_Ds shapes:", [[D.shape if isinstance(D, np.ndarray) else D for D in camera_intrinsic_params_Ds]])
        print("camera_frame_extrinsic_Rs shapes:", [[[R.shape if isinstance(R, np.ndarray) else R for R in Rs] for Rs in camera_frame_extrinsic_Rs]])
        print("camera_frame_extrinsic_Ts shapes:", [[[T.shape if isinstance(T, np.ndarray) else T for T in Ts] for Ts in camera_frame_extrinsic_Ts]])
        print("camera_frame_map_points_2d shapes:", [[[p.shape if isinstance(p, np.ndarray) else p for p in frame] for frame in camera_frame_map_points_2d]])
        print("camera_frame_map_points_2d_3d_index shapes:", [[idx if isinstance(idx, list) else [idx] for idx in frame] for frame in camera_frame_map_points_2d_3d_index])

        results = ba(map_points_3d, camera_intrinsic_params_Ks, camera_intrinsic_params_Ds, camera_frame_extrinsic_Rs, camera_frame_extrinsic_Ts, camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index)
        
        print("results", results)
        
        self.assertTrue(True), f"test_ba success"

if __name__ == '__main__':
    unittest.main()