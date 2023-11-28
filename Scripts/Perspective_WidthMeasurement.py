# python3 Perspective_WidthMeasurement.py
# Keep X value ( Width Measurement)
import numpy as np
import cv2
import math


# ArUco marker parameters and detection
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# Read the original image to detect the ArUco marker
img1 = cv2.imread("../Images/Pocket2-Trans.jpg")
img1 = cv2.rotate(img1, cv2.ROTATE_180) ### for 2

# Getting the ratio of pixel to mm
corners, _, _ = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=parameters)
print(corners)
print(corners[0][0][0][1])


# Check if any markers were detected
if len(corners) > 0:
    # Calculate the center of the detected marker
    center_x = int(np.mean([corners[0][0][j][0] for j in range(4)]))
    center_y = int(np.mean([corners[0][0][j][1] for j in range(4)]))
    A = (center_x, center_y)
    print(A)
else:
    raise ValueError("No markers detected in the image.")

aruco_perimeter = cv2.arcLength(corners[0], True)
pixel_cm_ratio = aruco_perimeter / 600
print(pixel_cm_ratio)

# Read the second image
img = cv2.imread('../Images/W3.jpg')
#img = cv2.rotate(img, cv2.ROTATE_180) ### for 2

# Find all back pixels in the image
y_coords, x_coords = np.where(np.all(img == [255, 255, 255], axis=-1)) # [255,255,255] white color
white_pixel_coords = list(zip(x_coords, y_coords))

# Sort white pixels by y and then x
sorted_white_pixel_coords = sorted(white_pixel_coords, key=lambda k: (k[1], k[0]))


# Filter and sort pixels with the same x value as A in ascending y order
A_x, A_y = A
pixels_with_same_y_as_A = sorted([(x, y) for x, y in sorted_white_pixel_coords if y == A_y], key=lambda k: k[0])

# Print the point with the smallest X value among pixels with same Y as A
if pixels_with_same_y_as_A:
    point_with_smallest_X = pixels_with_same_y_as_A[0]  # [0] is the first pixel after sorting by X
    print(f"Point with same Y as A and smallest X value: {point_with_smallest_X}")

    # Draw a line between A and point_with_smallest_X
    cv2.line(img1, A, point_with_smallest_X, (0, 0, 255), 2)  # Line in red color with thickness 2

    # Calculate the distance between the two points
    distance = np.linalg.norm(np.array(A) - np.array(point_with_smallest_X))
    print(f"Distance between A and the point with smallest X: {distance} pixels")
    print(f"Distance in cm: {distance/pixel_cm_ratio} cm")

else:
    print("No pixels found with the same Y value as A.")

# Optionally display the result
cv2.namedWindow('Image with Line', cv2.WINDOW_NORMAL)
cv2.imshow('Image with Line', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
