import cv2
import numpy as np

def onTrackbar( x ):
    pass

H_MIN = 0
S_MIN = 0
V_MIN = 0

H_MAX = 179
S_MAX = 255
V_MAX = 255

trackbarWindowName = 'Trackbar'

cap = cv2.VideoCapture( 0 )

cv2.namedWindow( trackbarWindowName )
cv2.createTrackbar( 'H_MIN', trackbarWindowName, 0, H_MAX, onTrackbar )
cv2.createTrackbar( 'H_MAX', trackbarWindowName, 0, H_MAX, onTrackbar )
cv2.createTrackbar( 'S_MIN', trackbarWindowName, 0, S_MAX, onTrackbar )
cv2.createTrackbar( 'S_MAX', trackbarWindowName, 0, S_MAX, onTrackbar )
cv2.createTrackbar( 'V_MIN', trackbarWindowName, 0, V_MAX, onTrackbar )
cv2.createTrackbar( 'V_MAX', trackbarWindowName, 0, V_MAX, onTrackbar )

while( 1 ):
    ret, frame = cap.read()
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )

    h_min = cv2.getTrackbarPos( 'H_MIN', trackbarWindowName )
    h_max = cv2.getTrackbarPos( 'H_MAX', trackbarWindowName )
    s_min = cv2.getTrackbarPos( 'S_MIN', trackbarWindowName )
    s_max = cv2.getTrackbarPos( 'S_MAX', trackbarWindowName )
    v_min = cv2.getTrackbarPos( 'V_MIN', trackbarWindowName )
    v_max = cv2.getTrackbarPos( 'V_MAX', trackbarWindowName )

    lowerLimit = np.array( [ h_min, s_min, v_min ] )
    upperLimit = np.array( [ h_max, s_max, v_max ] )

    mask = cv2.inRange( hsv, lowerLimit, upperLimit )
    res = cv2.bitwise_and( frame, frame, mask = mask )
    blurred = cv2.GaussianBlur(res, (5, 5), 0)
    kernel = np.ones( (5, 5), np.uint8 )
    eroded = cv2.erode( blurred, kernel, iterations = 1 )
    dilated = cv2.dilate(eroded, kernel, iterations = 1)
    gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours( frame, contours, -1, ( 0, 255, 0 ), 3 )
    
    if len( contours ) > 0:
        x,y,w,h = cv2.boundingRect( max( contours, key = cv2.contourArea ) )
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow( 'frame', frame )
    cv2.imshow( 'mask', mask )
    cv2.imshow( 'res', res )
    cv2.imshow( 'dilated', dilated )
    cv2.imshow( 'gray', gray )
    cv2.imshow( 'blurred', blurred )

    k = cv2.waitKey( 5 ) & 0xff

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()