import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#img = cv.imread('messi5.jpg')
img = cv.imread('nnpc.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (3,3,166,135)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
img[np.where((img == [0,0,0]).all(axis = 2))] = [204,122,0]
#cv.rectangle(img,(3,3),(166,135),(0,255,0),3)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
#plt.imshow(img),plt.colorbar(),plt.show()