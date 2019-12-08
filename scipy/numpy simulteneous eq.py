import numpy as np

A = np.array([[40, 20j], [30j, 60]])
B = np.array([100., 0.])
res = np.matmul( np.linalg.inv(A), B )
print( res )