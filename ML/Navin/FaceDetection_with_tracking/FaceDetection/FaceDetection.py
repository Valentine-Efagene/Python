# Source: Navin Kumar Manaswi - Deep Learning with Applications
# in python Chatbots,and facem Object...

import cv2
import dlib

#Create the tracker we will use to recognize face in different frames we get from the webcam
tracker = dlib.correlation_tracker()

#The Boolean variable we use to keep track whether we are using dlib tracker or not
trackingFace = 0

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
OUTPUT_SIZE_WIDTH = 700
OUTPUT_SIZE_HEIGHT = 600

capture = cv2.VideoCapture(0)
cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("base-image", 20, 200)
cv2.moveWindow("result-image", 640, 200)

cv2.startWindowThread()

rectangleColor = (0, 100, 255)

while(True):
    #Retrieve the largest image from the webcam
    rc, fullSizeBaseImage = capture.read()
    #Resize the image to 520, 420
    baseImage = cv2.resize(fullSizeBaseImage, (520, 420))

    #Check for keypress
    pressKey = cv2.waitKey(2) & 0xff
    if (pressKey == ord('Q') | pressKey == ord('q')):
        capture.release()
        cv2.destroyAllWindows()
        exit(0)

    if not trackingFace:
        resultImage = baseImage.copy()
        grayImage = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayImage, 1.3, 5)

        maxArea = 0
        x = y = w = h = 0

        for(_x, _y, _w, _h) in faces:
            if _w * _h > maxArea:
                x = _x
                y = _y
                w = _w
                h = _h
                maxArea = w * h

        if maxArea > 0:
            # Initialize the tracker
            tracker.start_track(baseImage,
                                dlib.rectangle(x-10, y - 20, x + w + 10, y + h + 20))
            trackingFace = 1

    if trackingFace:
        # Update the tracker and request information about the 
        # quality of the tracking update
        trackingQuality = tracker.update(baseImage)

        # If the tracking quality is good enough, determine the 
        # updated position of the tracked region and draw the rectangle
        if trackingQuality >= 9.0:
            tracked_position = tracker.get_position()
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            cv2.rectangle(resultImage, (t_x, t_y), 
                          (t_x + t_w, t_y + t_h), rectangleColor, 2)
            print("Tracking: ", t_x, " ", t_y)
        else:
            # If the quality of the tracking update is not good enough
            # for us (e.g. the face tracked moved out of the screen
            # we stop the tracking of the face and in the next loop we will find the largest
            # face in the image agin
            trackingFace = 0

    largeResult = cv2.resize(resultImage, 
                             (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
    cv2.imshow('base-image',baseImage)
    cv2.imshow('result-image', largeResult)