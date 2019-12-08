import numpy as np
import math
import sympy
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
from IPython.display import display_latex
from sympy import latex

sympy.init_printing()
np.set_printoptions(precision=3)

# Usage: display_equation('u_x', x)
def display_equation(idx, symObj):
    if(isinstance(idx, str)):
        eqn = '\\[' + idx + ' = ' + latex(symObj) + '\\]'
        display_latex(eqn, raw=True)
    else:
        eqn = '\\[' + latex(idx) + ' = ' + latex(symObj) + '\\]'
        display_latex(eqn, raw=True)
    return

def rotation(theta):
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
    return R

def createEquivRot(theta, K):
    R = np.zeros((3, 3))
    theta = (np.pi / 180) * theta
    
    R[0][0] = K[0][0] * K[0][0] * (1 - math.cos(theta)) + math.cos(theta)
    R[0][1] = K[0][0] * K[1][0] * (1 - math.cos(theta)) - K[2][0] * math.sin(theta)
    R[0][2] = K[0][0] * K[2][0] * (1 - math.cos(theta)) + K[1][0] * math.sin(theta)

    R[1][0] = K[0][0] * K[1][0] * (1 - math.cos(theta)) + K[2][0] * math.sin(theta)
    R[1][1] = K[1][0] * K[1][0] * (1 - math.cos(theta)) + math.cos(theta)
    R[1][2] = K[1][0] * K[2][0] * (1 - math.cos(theta)) - K[0][0] * math.sin(theta)

    R[2][0] = K[2][0] * K[2][0] * (1 - math.cos(theta)) - K[1][0] * math.sin(theta)
    R[2][1] = K[1][0] * K[2][0] * (1 - math.cos(theta)) + K[0][0] * math.sin(theta)
    R[2][2] = K[2][0] * K[2][0] * (1 - math.cos(theta)) + math.cos(theta)
    return R

def vplot2d(O, a, origin_label='O', tip_label='P'):
    dx = a[0][0] - O[0][0]
    dy = a[1][0] - O[1][0]

    head_length = 0.3
    vec_ab = [dx,dy]

    vec_ab_magnitude = math.sqrt(dx**2+dy**2)

    dx = dx / vec_ab_magnitude
    dy = dy / vec_ab_magnitude

    vec_ab_magnitude = vec_ab_magnitude - head_length

    ax = plt.axes()
    ax.set_aspect('equal', 'box')
    #ax.axis([-range, range, -range, range]) Sub for xlim and ylim

    ax.arrow(O[0][0], O[1][0], vec_ab_magnitude*dx, vec_ab_magnitude*dy, head_width=0.1, head_length=head_length, fc='lightblue', ec='black')
    plt.scatter(O[0][0],O[1][0],color='black')
    plt.scatter(a[0][0],a[1][0],color='black')

    ax.annotate(origin_label, (O[0][0]-0.4,O[1][0]),fontsize=14)
    ax.annotate(tip_label, (a[0][0]+0.3,a[1][0]),fontsize=14)
    return

def vplot3d(O, P, origin_label='O', tip_label='P'):
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

# plot a frame 
def fplot3d(B, T = np.identity(4), origin_label='O', tip_label='P', x_range = [-2, 2], y_range = [-2, 2], z_range=[-2, 2]):
    mpl.rcParams['legend.fontsize'] = 10
    m = 1

    ax = fig.gca(projection='3d')
    R = T[:3,:3]
    
    B = np.expand_dims(T[:3,3], axis=0).T
    
    Bx = homTrans(T, np.array([[m, 0, 0]]).T)
    By = homTrans(T, np.array([[0, m, 0]]).T)
    Bz = homTrans(T, np.array([[0, 0, m]]).T)
    
    vplot3d(B, Bx, origin_label='B', tip_label='x')
    vplot3d(B, By, origin_label='B', tip_label='y')
    vplot3d(B, Bz, origin_label='B', tip_label='z')
    
    ax.scatter(B[0][0], B[1][0], B[2][0], marker='x')
    ax.axis('equal')
    ax.set_aspect('equal')
    ax.auto_scale_xyz(x_range, y_range, z_range)
    #ax.pbaspect= [20, 2, 2]
    ax.legend()
    return

def homTrans(T, A):
    R = T[:3,:3]
    return (R @ A) + np.expand_dims(T[:3,3], axis=0).T

#tVec (translation vector) is a column vector, e.g. tMat = np.array([[10],[5],[0]])
# r is either a 3x3 rotation transformation matrix or an angle in degrees
def createHomTrans(r, tVec):
    tVec = np.vstack((tVec, [1]))
    T = np.zeros((4, 4))
        
    if (isinstance(r, np.ndarray)):
        T = np.vstack((r, np.array([0, 0, 0])))
    else:
        T = np.vstack((rotation(r), np.array([0, 0, 0])))
    
    T = np.append(T, tVec, axis=1)
    return T

'''
# Example 2.8
m = 2 # This is a magnitude scalar to make K visible in the plot
O = np.zeros((3, 1))
R = np.zeros((3, 3))
K = np.array([[0.7070, 0.7070, 0]]).T
tVec = np.array([[1.0, 2.0, 3.0]]).T

R = createEquivRot(30, K)
T = createHomTrans(R, O)

A = np.zeros((3, 1))
fig = plt.figure()
# Origin frame
fplot3d(A, tip_label='B')
vplot3d(A, K * m, origin_label='A', tip_label='K')
fplot3d(A, T, tip_label='B',
        x_range = [-2, 2], y_range = [-2, 2], z_range = [-2, 2])
plt.show()
'''


#Example 2.9
O = np.zeros((3, 1))
R = np.zeros((3, 3))
K = np.array([[0.7070, 0.7070, 0]]).T
tVec = np.array([[1.0, 2.0, 3.0]]).T
P = np.array([[1.0, 2.0, 3.0]]).T

R = createEquivRot(30, K)
BTB = createHomTrans(0, -tVec)
ATA = createHomTrans(0, tVec)
ATB = createHomTrans(R, O)

T = ATA @ ATB @ BTB
print(T)

m = 2 # This is a magnitude scalar to make K visible in the plot
A = np.zeros((3, 1))
P *= 0.5

S = P + K

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(P[0][0], P[1][0], P[2][0], marker='P')

fplot3d(A, tip_label='B')
vplot3d(P, S, origin_label='A', tip_label='K')
fplot3d(A, T, tip_label='B',
        x_range = [-2, 2], y_range = [-2, 2], z_range = [-2, 2])
plt.show()
