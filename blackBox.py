import numpy as np

# Define a function
def square(x):
    return x**2

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Use numpy.vectorize to apply the function to each element
square_fn = np.vectorize(square)
result = square_fn(arr)

print("Original array:", arr)
print("Result after applying the function:", result)
