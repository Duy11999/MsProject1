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
img = cv2.imread("../Images/20230629_142615.jpg")
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

if len(contours) > 0:
    # Find the index of the largest contour
    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    largest_contour = contours[largest_contour_index]

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Check if the contour has four vertices (a rectangle)
    if len(approx) == 4:
        # Draw the rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        corners = approx.reshape(-1, 2)
        object_width = w / pixel_cm_ratio
        object_height = h / pixel_cm_ratio
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -2)
        cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
        cv2.imshow("img2", img)
        print(object_height)
        print(object_width)

        cv2.putText(img, "Width {}".format(round(object_width, 3)), (int(x), int(y) - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 216, 235), 2)
        cv2.putText(img, "Height {}".format(round(object_height, 3)), (int(x), int(y) + 20), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 216, 235), 2)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)
