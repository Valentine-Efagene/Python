import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

theta = 30
theta = (np.pi / 180) * theta
R = np.zeros((3, 3))

R[0][0] = np.cos(theta)
R[0][1] = -np.sin(theta)
R[0][2] = 0
R[1][0] = np.sin(theta)
R[1][1] = np.cos(theta)
R[1][2] = 0
R[2][0] = 0
R[2][1] = 0
R[2][2] = 1

BP = np.array([[0.0], [2.0], [0.0]])
# print(R.dot(BP))
AP = R @ BP # New operator since python 3.5

def plot3d(O, P, origin_label='O', tip_label='P'):
    mpl.rcParams['legend.fontsize'] = 10

    ax = fig.gca(projection='3d')
    x = np.linspace(O[0][0], P[0][0], 100)
    y = np.linspace(O[1][0], P[1][0], 100)
    z = np.linspace(O[2][0], P[2][0], 100)
    
    ax.plot(x, y, z, label=tip_label)
    ax.scatter(P[0][0], P[1][0], P[2][0], marker='x')
    ax.axis('equal')
    ax.legend()
    return

O = [[0],[0], [0]]
fig = plt.figure()
plot3d(O, AP, tip_label='A')
plot3d(O, BP, tip_label='B')
plt.show()
