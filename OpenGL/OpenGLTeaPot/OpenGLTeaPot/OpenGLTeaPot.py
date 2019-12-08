from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys

def draw():
    glutWireTeapot(0.5)
    glFlush()

glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
glutInitWindowSize(200, 200)
glutInitWindowPosition(100, 100)
glutCreateWindow("My first OGL program")
glutDisplayFunc(draw)
glutMainLoop()