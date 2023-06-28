# python3 perspectiveTrans.py
import numpy as np
import cv2
import math

# Read input and get corners of aruco
img = cv2.imread('../Images/Slab2R.jpg')
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

hh, ww = img.shape[:2]

# Specify input coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x, y
input = np.float32(corners[0][0])  # Access the inner array of corners
print(input)

# Get top and left dimensions and set them as the width and height of the output rectangle
width = round(math.hypot(input[0, 0] - input[1, 0], input[0, 1] - input[1, 1]))
height = round(math.hypot(input[0, 0] - input[3, 0], input[0, 1] - input[3, 1]))
print("width:", width, "height:", height)

# Set upper left coordinates for output rectangle
x = input[0, 0]
y = input[0, 1]

# Specify output coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x, y
output = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

# Compute perspective matrix
matrix = cv2.getPerspectiveTransform(input, output)
print(matrix)

# Calculate the transformed coordinates of all four corners of the original image
original_corners = np.float32([[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]])
transformed_corners = cv2.perspectiveTransform(np.array([original_corners]), matrix)[0]

# Find the minimum and maximum x and y values of the transformed coordinates
min_x = np.min(transformed_corners[:, 0])
max_x = np.max(transformed_corners[:, 0])
min_y = np.min(transformed_corners[:, 1])
max_y = np.max(transformed_corners[:, 1])

# Calculate the width and height of the bounding box for the transformed image
bbox_width = int(np.ceil(max_x - min_x))
bbox_height = int(np.ceil(max_y - min_y))

# Calculate the translation matrix to shift the bounding box to positive coordinates
translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

# Apply the translation matrix to the perspective transformation matrix
final_matrix = translation_matrix.dot(matrix)

# Do perspective transformation using the final matrix
output_size = (bbox_width, bbox_height)
imgOutput = cv2.warpPerspective(img, final_matrix, output_size, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Save the warped output
cv2.imwrite("slab6(1).jpg", imgOutput)

# Show the result
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", imgOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
