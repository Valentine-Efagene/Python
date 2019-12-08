import cv2
import numpy as np

path = "numbers.png"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (7,7), 0)
ret, thbw = cv2.threshold(bw, 200, 255, cv2.THRESH_BINARY_INV)
thbw = cv2.erode(thbw, np.ones((1,1), np.uint8), iterations = 2)
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("img", img)
cv2.imshow("bw", bw)
cv2.imshow("thbw", thbw)
cv2.waitKey()