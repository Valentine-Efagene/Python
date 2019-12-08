import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Mouth.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    a = 0

    if len( faces ) > 0:
        for (x_,y_,w_,h_) in faces:
            if x_ * y_ > a:
                a = x_ * y_
                x = x_
                y = y_
                w = w_
                h = h_

        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    
        if len( eyes ) > 0:
            a = 0

            for (ex_,ey_,ew_,eh_) in eyes:
                if ey_ > a:
                    a = ey_
                    ex = ex_
                    ey = ey_
                    ew = ew_
                    eh = eh_

            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
            
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()