from typing import Type, List, Sequence, TypedDict, Dict, Union, Tuple
import numpy as np
import jax.numpy as jnp

NestListOfNDArrays = List[Union[np.ndarray, "NestListOfNDArrays"]]
NestListsOfJNDArrays = List[Union[jnp.ndarray, "NestListsOfJNDArrays"]]

class NDArrayMetaData(TypedDict):
    shape: Tuple[int, ...]
    dtype: str

class CodecMetaData(TypedDict):
    length: int
    type: str
    child_meta: List[Union['NDArrayMetaData', 'CodecMetaData']]
    total_size: int
    

"""
JAXNestedArrayCodec

An abstraction to allow us to encode / decode our params format which is a list of nested lists of any depth where
the lowest element structure is an arbitrary nd array

The idea is while we are serialising we generate a metadata object that
can be used to pack the data into a jax 1D array. We also implement a deserialise method
so that we can go from a 1D array back to nested lists containing jax array of the correct shape as determined by the
metadata.

"""



"""
High level idea... take nested lists of lists of any length with bottom element of ndarrays
and flatten them to a single JAX 1d array with an accompanying metadata object which allows for reconstruction
of the original structure
"""
def serialise(params:NestListOfNDArrays, top_level=True) -> Tuple[jnp.ndarray, List[CodecMetaData]]:
    #print("calling serialise params", params, type(params))
    if top_level is True:
        #print("00000000000000000000000000000000000000000")
        #rvecs = params[3]
        #tvecs = params[4]
        #print("mapoints3d", params[0])
        #print("Ks", params[1])
        #print("Ds", params[2])
        #print("rvecs",rvecs)
        #print("tvecs",tvecs)
        #print("converted_camera_frame_map_points_2d", params[5])
        
        #[map_points_3d, converted_camera_intrinsic_Ks, converted_camera_intrinsic_Ds, converted_camera_frame_extrinsic_rvecs, converted_camera_frame_extrinsic_tvecs, converted_camera_frame_map_points_2d]
        pass
    #if isinstance(params, np.ndarray):
    #    print("np.ndarray")
    data = []
    meta = []
    if isinstance(params, np.ndarray):
        shape = params.shape
        meta.append({"shape": shape, "type": "ndarray"})
        data.append(params.flatten())
    if isinstance(params, jnp.ndarray):
        shape = params.shape
        meta.append({"shape": shape, "type": "ndarray"})
        data.append(np.asarray(params).flatten())
    elif isinstance(params, list):
        length = len(params)
        this_level_meta = {"length": length, "type": "list", "child_meta": [], "total_size": 0}
        next_level_data = []
        next_level_meta_data = []
        #print("params", params, type(params[0]))
        #print("]]]]")
        for i in params:
            new_data, new_metadata = serialise(i, False)
            next_level_meta_data.extend(new_metadata)
            next_level_data.extend(new_data)
        this_level_meta["child_meta"] = next_level_meta_data
        #print("next_level_data", next_level_data)
        this_level_meta["total_size"] = len(np.concatenate(next_level_data))
        data.extend(next_level_data)
        meta.append(this_level_meta)
    
    if top_level is True:
        data = np.concatenate(data)
        data = jnp.asarray(data)
        
    return data, meta

"""
Take the 1D jax data array and reconstruct the data into the right shape of nested lists of jax arrays
"""
def deserialise(data: jnp.array, metadata: List[CodecMetaData], top_level=True) -> NestListsOfJNDArrays:
    #print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii desiralise", data.shape)
    index = 0  # Track the current position in the flattened data array
    result = []
    # Iterate over metadata to reconstruct the nested structure
    #print("metadata", metadata)
    for meta in metadata:
        #print("meta", meta)
        if meta["type"] == "ndarray":
            #print("got an nd array")
            # If the metadata indicates an ndarray, extract the shape information
            shape = meta["shape"]
            # Slice the data array to get the flattened ndarray
            sliced_data = data[index:index + np.prod(shape)].reshape(shape)
            # Convert the sliced data to a JAX array
            jax_array = jnp.asarray(sliced_data)
            # Update the index
            index += np.prod(shape)
            # Append the flattened ndarray to the result list
            result.append(jax_array) # .flatten()            
        elif meta["type"] == "list":
            #print("got a list")
            # If the metadata indicates a list, extract the total size information
            total_size = meta["total_size"]
            # Slice the data array to get the flattened sublist
            sublist_data = data[index:index + total_size]
            # Convert the sliced data to a JAX array
            sublist_array = jnp.asarray(sublist_data)
            # Update the index
            index += total_size
            # Recursively call deserialise to process the nested list
            sublist = deserialise(sublist_array, meta["child_meta"], False)
            # Append the sublist to the result list
            result.append(sublist)
    #print("got to return", result)   
    if top_level is True:
        #print("top got to return", result)   
        ahh = result[0]
        #print("got ahh", ahh)
        return ahh
    
    return result
"""
class JaxOptParamCodex():
    array_codec = None
    def __init__(self) -> None:
        self.array_codec = JAXNestedArrayCodec()
    
    def pack(self, params: Dict[str, NestListOfNDArrays]) -> Tuple[jnp.ndarray, List[CodecMetaData], List[str]]:
        keys, values = zip(*params.items())
        keys = list(keys)
        values = list(values)
        data, metadata = self.array_codec.serialise(values)
        return data, metadata, keys
    
    def unpack(self, data: jnp.ndarray, metadata: List[CodecMetaData], keys: List[str]) -> Dict[str,NestListsOfJNDArrays]:
        original_structure = self.array_codec.deserialise(data, metadata)
        ret = {}
        for idx, k in enumerate(keys):
            ret[k] = original_structure[idx]
        return ret




"""

"""
print("example_data_to_pack", example_data_to_pack)


NestListOfNDArrays = List[Union[np.ndarray, "NestListOfNDArrays"]]
OptParams = Dict[str, NestListOfNDArrays]




"""

"""
OptNestedNDArraySourceValue = Union[np.ndarray, "OptNestedNDArraySourceDict"]
OptNestedNDArraySourceDict = Dict[str, OptNestedNDArraySourceValue]

OptNestedJNDArrayDestinationValue = Union[jnp.ndarray, "OptNestedJNDArrayDestinationValue"]
OptNestedJNDArrayDestinationDict = Dict[str, OptNestedJNDArrayDestinationValue]


class OptCodecNDArrayMetaDataElement(TypedDict):
    shape: Tuple[int,...]
    dtype: str
    
OptCodecNDArrayMetaDataElement =  Union[OptCodecNDArrayMetaDataElement, List[OptCodecNDArrayMetaDataElement], List["OptCodecNDArrayMetaData"]]
OptCodecNDArrayMetaData = List[OptCodecNDArrayMetaDataElement]

def serialise(params: OptNestedNDArraySourceDict) -> Tuple[OptNestedJNDArrayDestinationValue, OptCodecNDArrayMetaData]:
    metadata: OptCodecNDArrayMetaData = []
    flat_params_np: Union[OptNestedJNDArrayDestinationValue, OptNestedNDArraySourceValue] = []
    
    if isinstance(params, dict):
        for k, v in enumerate(params):
            #metadata[k]
            nested_flat_params_np, nested_meta_data = serialise(v)
            metadata.extend(nested_meta_data)#
            flat_params_np.extend(nested_flat_params_np)
    elif isinstance(params)
            
            
    elif isinstance(params, np.ndarray):

        pass
    
    return (jnp.ndarray(flat_params_np), metadata)
    

def deserialise(input: jnp.ndarray, metadata: OptCodecNDArrayMetaData) ->  OptNestedJNDArrayDestinationDict:
    pass
    
"""