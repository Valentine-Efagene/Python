import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageToArrayConverter import ImageToArrayConverter

itacTrain = ImageToArrayConverter('calibri_12_caps_no_punctuations.png')
x_vals_train = itacTrain.getImageAsArray()

threshold = itacTrain.getThresholdedImage()
contour = itacTrain.getContourImage()

print(itacTrain.getNumberOfDigits())

plt.subplot(121),plt.imshow(contour),plt.title('Contours')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(threshold),plt.title('Extracted')
plt.xticks([]), plt.yticks([])
plt.show()