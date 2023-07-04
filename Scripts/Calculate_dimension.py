# python3 Calculate_dimension.py
import cv2
from object_detector import *
import numpy as np


#Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


#Load Object Detector
detector = HomogeneousBgDetector()
#Load the image
img = cv2.imread("../Images/a(1).jpg")
#Get Aruco marker
corners,_,_ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)
# Pixel to mm ratio
pixel_cm_ratio = aruco_perimeter / 400

#Draw Polygon around marker
int_conners = np.intp(corners)
cv2.polylines(img,int_conners,True, (235,255,0),3)
contours = detector.detect_objects(img)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow("image1",img)


#Draw objects boundaries:
for cnt in contours:
    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x,y), (w,h), angle = rect # width and height in pixel of object on image

    #Get Width and Height of Objects by applying Ration pixel to cm
    object_width = w/pixel_cm_ratio
    object_height = h/pixel_cm_ratio

    #Display rectangle
    box = cv2.boxPoints(rect)
    box = np.intp(box) # array contains 4 coordinates of 4 corners
    # Draw polygons
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -2)
    cv2.polylines(img, [box], True, (255, 126, 0), 3)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow("img2", img)
    print(object_height)
    print(object_width)

    cv2.putText(img, "Width {}".format(round(object_width,3)), (int(x),int(y)-20), cv2.FONT_HERSHEY_PLAIN,1,(255,216,235),2)
    cv2.putText(img, "Height {}".format(round(object_height,3)), (int(x),int(y)+20), cv2.FONT_HERSHEY_PLAIN,1,(255,216,235),2)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)