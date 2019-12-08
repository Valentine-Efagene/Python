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

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
xA = np.linspace(0, AP[0][0], 100)
yA = np.linspace(0, AP[1][0], 100)
zA = np.linspace(0, AP[2][0], 100)

xB = np.linspace(0, BP[0][0], 100)
yB = np.linspace(0, BP[1][0], 100)
zB = np.linspace(0, BP[2][0], 100)

ax.plot(xA, yA, zA, label='vector A')
ax.plot(xB, yB, zB, label='vecfor B')
ax.scatter(AP[0][0], AP[1][0], AP[2][0], marker='x')
ax.scatter(BP[0][0], BP[1][0], BP[2][0], marker='x')
ax.legend()

plt.show()