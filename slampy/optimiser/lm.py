from slampy.optimiser.codec import serialise, deserialise
from jax import grad, hessian, numpy as jnp
import jax

"""
trying to emulate this syntax

    result = minimize(loss_function, initial_params,
                    args=(camera_frame_map_points_2d, camera_frame_map_points_2d_3d_index),
                    method='lm')
                    
    def loss_function(params_to_optimise, ...args):

def example_loss_functions(params_to_optimise, *args, metadata=None, **kwargs):
    unpacked_params = param_codex.unpack(params_to_optimise, metadata)
    pass

"""

"""
| Parameter | Description | Lower Bound (Units) | Upper Bound (Units) |
| --- | --- | --- | --- |
| cx | Principal point x-coordinate | 0 | image_width (pixels) |
| cy | Principal point y-coordinate | 0 | image_height (pixels) |
| fx | Focal length along x-axis | 10 mm | 100 mm |
| fy | Focal length along y-axis | 10 mm | 100 mm |
| skew | Skew parameter | -1 | 1 |
| k1 | Radial distortion coefficient (1st order) | -0.2 | 0.05 |
| k2 | Radial distortion coefficient (2nd order) | -0.1 | 0.03 |
| p1 | Tangential distortion coefficient (1st order) | -0.001 | 0.001 |
| p2 | Tangential distortion coefficient (2nd order) | -0.001 | 0.001 |
| k3 | Radial distortion coefficient (3rd order) | -0.0001 | 0.005 |
| k4 | Radial distortion coefficient (4th order) | -0.00001 | 0.00001 |
| k5 | Radial distortion coefficient (5th order) | -0.00001 | 0.00001 |
| k6 | Radial distortion coefficient (6th order) | -0.00001 | 0.00001 |
| aspect | Aspect ratio (typically fy/fx) | 0.1 | ∞ |
"""


def lm(a):
    return jnp.sum((a)**2) # a is data b is some model 

def example_loss_functions(params, *args):
    params # unpack
    
    # compare projected 2d and 3d and take old - new 2d point distances
    pass
    #Projection error


def minimise(udf_loss_function, initial_params_estimate, args=(), epoch=3, damping_factor=1000):
    print("starting minimise")
    previous_loss = jnp.inf
    previous_udf_additional_return_values = None
    params = initial_params_estimate
    for i in range(epoch):
        print("epoch", i, "-----------------------------------------------------------------------")
        
        flat_params, flat_params_meta = serialise(params)
        


        # i think we need to wrap this bit so that we use the lm method here
        result = udf_loss_function(params, *args)
        
        
        if isinstance(result, tuple):
            loss = result[0]
            udf_additional_return_values = result[1:] if len(result) > 1 else ()
        else:
            # If result is a single value, convert it to a tuple
            loss = result
            udf_additional_return_values = ()
        
        
        # Change in udf output wrt the current params estimate
        # Our loss_function returns a scalar, over the space of all input params, it act as a scalar field.
        # The gradient computes the change of functions output value wrt to the input param, and thus give us a vector of first order derivatives of the scalar field.
        # ND to 1D differentiation is the grad function
        
        # also here maybe we need the lm method again
        udf_grad_wrt_params = jax.grad(lambda x: udf_loss_function(deserialise(x, flat_params_meta), *args)) #[0]
        #udf_grad_wrt_params = jax.grad(lambda x: (udf_loss_function(deserialise(x, flat_params_meta), *args),) if isinstance(udf_loss_function(deserialise(x, flat_params_meta), *args), tuple) else (udf_loss_function(deserialise(x, flat_params_meta), *args),))

        #udf_grad_wrt_params = jax.grad(lambda x: udf_loss_function(deserialise(x, flat_params_meta), *args)[0])

        # Find the 1st order udf wrt params
        udf_jac = jax.jit(udf_grad_wrt_params)(flat_params)
        
        # Find the 2nd order udf jac wrt params
        # multi variate functions local curvature 
        # JT J
        
         
         
        #udf_hessian = jax.jit(udf_loss_function)(flat_params)
        udf_hessian = jax.jit(jax.hessian(lambda params: udf_loss_function(deserialise(params, flat_params_meta), *args)))(flat_params)

        #udf_hessian = jax.jit(jax.hessian(lambda x: udf_loss_function(deserialise(x, flat_params_meta), *args)))(flat_params)
        
        
        # damping facor
        damping_matrix = jnp.eye(len(flat_params))*damping_factor

        # compute change
        # d = (JT J + lambda I)^-1 * JT(observed_data - model_prediction)
        # make sure we decend the gradient
        # lm update method
        change_in_params = jnp.linalg.solve(udf_hessian + damping_matrix, -udf_jac.flatten())
        
        print("----------------------------------------------------")
        
        print("flat_params_meta", flat_params_meta)
        print("change_in_params", change_in_params)
        print("params", params)
        print("flat params", flat_params)
        
        #params += change_in_params
        
        print("----------------------------------------------------")

        
        flat_params += change_in_params / 1000000
        print("flat_params_updated", flat_params, flat_params.shape)
        
        params = deserialise(flat_params, flat_params_meta)
        print("params final", params)
        print("flat_params_meta", flat_params_meta)
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        previous_loss = loss        
        #print("got loss", previous_loss)
        previous_udf_additional_return_values = udf_additional_return_values

    return params, previous_loss, *previous_udf_additional_return_values
