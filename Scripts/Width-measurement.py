# python3 Width-measurement.py
import cv2
import numpy as np
from collections import defaultdict

parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

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

### Edge Detection ###

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
img_median = cv2.medianBlur(img_bilateral, 5)
img_blur = cv2.GaussianBlur(img_median, (3, 3), 0)

sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

y, x = np.where(sobel_y_abs > 0)
edge_points = list(zip(x, y))

points_by_y = defaultdict(list)
for point in edge_points:
    x_val, y_val = point
    points_by_y[y_val].append(point)

# Find the top 5 groups based on the number of points they contain
top_5_groups = sorted(points_by_y.items(), key=lambda item: len(item[1]), reverse=True)[:5]

# Among the top 5, get the group with the highest Y value
highest_y_among_top_5 = max(top_5_groups, key=lambda item: item[0])

# Print this group
print(f"Among the top 5 groups, Y={highest_y_among_top_5[0]} has the highest Y value and contains points: {highest_y_among_top_5[1]}")

# Display the Sobel Y edge detected image
cv2.namedWindow('Sobel Y Edges', cv2.WINDOW_NORMAL)
cv2.imshow('Sobel Y Edges', sobel_y_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()

