from codec import serialise, deserialise#JAXNestedArrayCodec, JaxOptParamCodex
import unittest
from jax import numpy as jnp
import numpy as np


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

class TestSerialisationDeserialisation(unittest.TestCase):    
    def test_serialisation_deserialisation(self):
        # Test multiple cases here
        
        array1 = np.asarray([1, 2, 3, 4])
        array2 = np.asarray([5, 6, 7])
        array3 = np.asarray([8, 9, 10])
        ndarray = np.asarray([array3, array2])
        complex_example = [array1, array2, [[array1, array2], array1], ndarray]

        test_cases = [
            (np.asarray([1, 2, 3, 4]), "Test Case 1"),
            (np.asarray([5, 6, 7]), "Test Case 2"),
            ([np.asarray([1, 2, 3, 4]), np.asarray([5, 6, 7])], "Test Case 3"),
            (complex_example, "Test Case 4")
            # Add more test cases as needed
        ]
        
        #codec = JAXNestedArrayCodec()

        for data, name in test_cases:
            with self.subTest(name=name):
                flat_data, metadata = serialise(data) #codec.
                reconstructed_data = deserialise(flat_data, metadata) #codec.
                
                print("Serialise Test %s" % (name))
                print("Serialise input data", data)
                print("Serialise output flat_data", flat_data)
                print("Serialise output metadata", metadata)
                print("~~~~~")
                print("Deserialise Test %s" % (name))
                print("Deserialise input data", data)
                print("Deserialise input metadata", metadata)
                print("Deserialise output reconstructed_data", reconstructed_data)
                print("--------------------------------------------------------------------")
                
                
                self.assertTrue(recursive_array_equal(data, reconstructed_data), f"Failed for {name}")
        
        # Ensure we can pack the same structure twice before unpacking it
        complex_example_copy = complex_example.copy()
        flat_data_1, metadata = serialise(complex_example)
        flat_data_2, metadata = serialise(complex_example)
        
        data_1 = deserialise(flat_data_1, metadata)
        data_2 = deserialise(flat_data_2, metadata)
        
        self.assertTrue(recursive_array_equal(complex_example, data_1), f"Failed for by ref check 1")
        self.assertTrue(recursive_array_equal(complex_example_copy, data_1), f"Failed for by ref check 1")
        self.assertTrue(recursive_array_equal(complex_example, data_2), f"Failed for by ref check 3")
        self.assertTrue(recursive_array_equal(complex_example_copy, data_2), f"Failed for by ref check 4")
        
        #self.assertTrue(recursive_array_equal(flat_data_2, data_2), f"Failed for by ref check 2")
        #self.assertTrue(recursive_array_equal(flat_data_1, complex_example), f"Failed for by ref check 3")
        #self.assertTrue(recursive_array_equal(flat_data_2, complex_example), f"Failed for by ref check 5")
        
        
        
        
        #self.assertTrue(recursive_array_equal(data, reconstructed_data), f"Failed for {name}")

    """def test_packing_and_unpacking(self):
        # Test multiple cases here
        
        array1 = np.asarray([1, 2, 3, 4])
        array2 = np.asarray([5, 6, 7])
        array3 = np.asarray([8, 9, 10])
        ndarray = np.asarray([array3, array2])
        
        test_cases = [
            ({"k1": array1, "k2": array2, "k3": [[array1, array2], array1], "k4": ndarray}, "Test Case 1")
            # Add more test cases as needed
        ]
        
        codec = JaxOptParamCodex()

        for data, name in test_cases:
            with self.subTest(name=name):
                print("data", data)
                flat_data, metadata, keys = codec.pack(data)
                reconstructed_data = codec.unpack(flat_data, metadata, keys)
                
                print("Pack Test %s" % (name))
                print("Pack input data", data)
                print("Pack output flat_data", flat_data)
                print("Pack output metadata", metadata)
                print("Pack output keys", keys)
                print("~~~~~")
                print("Unpack Test %s" % (name))
                print("Unpack input flat_data", data)
                print("Unpack input metadata", metadata)
                print("Unpack input keys", keys)
                print("Unpack output reconstructed_data", reconstructed_data)
                print("--------------------------------------------------------------------")
                
    """


if __name__ == '__main__':
    unittest.main()