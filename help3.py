import jax.numpy as jnp

# Example 2D array
data_2d = jnp.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

# Create a mask where the first element of each row is equal to 4
mask = data_2d[:, 0] == 4

# Print the mask
print(mask)