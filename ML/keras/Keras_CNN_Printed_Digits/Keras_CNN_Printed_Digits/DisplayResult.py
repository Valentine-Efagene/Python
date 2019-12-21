import cv2
import numpy as np
import random

class DisplayResult(object):
    """description of class"""
    def __init__(self, src, vals):
        self.src = src
        font = cv2.FONT_HERSHEY_SIMPLEX

        path = self.src
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #bw = cv2.GaussianBlur(bw, (7,7), 0) # Including this line usually ends up connecting characters, but helps when characters erronously start getting broken into multiple
        ret, thbw = cv2.threshold(bw, 200, 255, cv2.THRESH_BINARY_INV)
        thbw = cv2.erode(thbw, np.ones((1,1), np.uint8), iterations = 2)
        cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Using RETR_EXTERNAL ensures 
        #only the highest heirarchy are taken. This ensures letters like 'R' are not seen as two letters

        rectangles = []

        for c in cntrs:
            r = x,y,w,h = cv2.boundingRect(c)
            a = cv2.contourArea(c)
            b = (img.shape[0]-3) * (img.shape[1] - 3)

            is_inside = False
            
            for q in rectangles:
                if self.inside(r, q):
                    is_inside = True
                    break
            if not is_inside:
                if not a == b:
                    rectangles.append(r)
     
        self.threshHoldedImage = thbw
        self.contourImage = img
        self.numberOfDigits = len(rectangles)
        self.x_vals_test = np.zeros( (self.numberOfDigits, 784) )
        i = 0
        font = cv2.FONT_HERSHEY_SIMPLEX 
        fontScale = 0.5
        thickness = 1
        b, g, r = 255, 0, 0
        above = 5

        for rec in rectangles:
            b = random.randint(0, 250)
            g = random.randint(0, 250)
            r = random.randint(0, 250)
            color = (b, g, r)
            x,y,w,h = self.wrap_digit(rec)
            cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), color, 2)
            self.contourImage = cv2.putText(img, str(int(vals[i])), (int(x), int(y - above)), font, fontScale, color, thickness, cv2.LINE_AA) 
            #cv2.line(self.contourImage, (int(x), int(y - 20)), (int(x), int(y - 10)), color, thickness)
            roi = thbw[int(y):int(y+h), int(x):int(x+w)]
            res = cv2.resize(roi, (28, 28))
            self.x_vals_test[i] = res.ravel()
            i += 1

    def inside(self, r1, r2):
        x1,y1,w1,h1 = r1
        x2,y2,w2,h2 = r2
        if (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and (y1+h1 < y2 + h2):
            return True
        else:
            return False

    def wrap_digit(self, rect):
        x, y, w, h = rect
        padding = 5
        hcenter = x + w/2
        vcenter = y + h/2
        
        if (h > w):
            w = h
            x = hcenter - (w/2)
        else:
            h = w
            y = vcenter - (h/2)
        return (x-padding, y-padding, w+padding, h+padding)

    def getImageAsArray(self):
            return self.x_vals_test

    def getContourImage(self):
            return self.contourImage

    def getGaussianBlur(self):
            return self.contourImage

    def getThresholdedImage(self):
            return self.threshHoldedImage
    def getNumberOfDigits(self):
        return self.numberOfDigits
