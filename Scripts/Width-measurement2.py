# python3 Width-measurement2.py
import cv2
import numpy as np
from object_detector import *

parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = HomogeneousBgDetector()

# Read the original image
img = cv2.imread("../Images/pocket2.jpg")

# Getting ratio pixel and mm
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Check if any markers were detected
if len(corners) > 0:
    # Calculate the center of the detected marker
    center_x = int(np.mean([corners[0][0][j][0] for j in range(4)]))
    center_y = int(np.mean([corners[0][0][j][1] for j in range(4)]))
    A = (center_x, center_y)  # Center of the first detected marker
else:
    raise ValueError("No markers detected in the image.")

aruco_perimeter = cv2.arcLength(corners[0], True)
pixel_cm_ratio = aruco_perimeter / 600
print(pixel_cm_ratio)

### Edges Detection
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Bilateral Filter to reduce noise but retain edges
img_bilateral = cv2.bilateralFilter(img_gray, 3, 65, 65)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_bilateral, (3, 3), 0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=75, threshold2=200)

# Display the Canny edge detected image
cv2.namedWindow('Canny Edges', cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)

# Get the coordinates of all edge pixels
y, x = np.where(edges == 255)  # returns the y and x coordinates respectively
edge_points = list(zip(x, y))

# Filter out the points with the same x-coordinate as point A
same_x_points = [pt for pt in edge_points if pt[0] == A[0]]

# If there are 1 or more points, draw a line between A and the point with the highest y value
if same_x_points:
    end_point = max(same_x_points, key=lambda pt: pt[1])  # gets the point with the lowest y value
    cv2.line(img, A, end_point, (0, 255, 0), 2)

    # Calculate the length of the line in pixels using the Euclidean distance formula
    length = np.sqrt((end_point[0] - A[0])**2 + (end_point[1] - A[1])**2)
    length1 = length/pixel_cm_ratio
    print(f"Length of the line: {length1} mm")

# Display the image with the drawn line
cv2.namedWindow('Image with Line', cv2.WINDOW_NORMAL)
cv2.imshow('Image with Line', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

