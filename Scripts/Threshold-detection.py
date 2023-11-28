# python3  Threshold-detection.py (pixel value)
# Width detection based SAM
# env Test1
import numpy as np
import cv2

# ArUco marker parameters and detection
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# Read the original image to detect the ArUco marker
img1 = cv2.imread("../Images/Pocket2.jpg")

# Getting the ratio of pixel to mm
corners, _, _ = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=parameters)

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
img = cv2.imread('../Images/W22.jpg')

# Find all back pixels in the image
y_coords, x_coords = np.where(np.all(img == [255, 255, 255], axis=-1)) # [255,255,255] white color
white_pixel_coords = list(zip(x_coords, y_coords))

# Sort white pixels by y and then x
sorted_white_pixel_coords = sorted(white_pixel_coords, key=lambda k: (k[1], k[0]))

# Filter and sort pixels with the same x value as A in ascending y order
A_x, A_y = A
pixels_with_same_x_as_A = sorted([(x, y) for x, y in sorted_white_pixel_coords if x == A_x], key=lambda k: k[1])


# Print the point with the highest Y value among pixels with same X as A
if pixels_with_same_x_as_A:
    point_with_highest_Y = pixels_with_same_x_as_A[0] # [-1] is the lastest pixel
    print(f"Point with same X as A and highest Y value: {point_with_highest_Y}")

    # Draw a line between A and point_with_highest_Y
    cv2.line(img1, A, point_with_highest_Y, (0, 0, 255), 2)  # Line in red color with thickness 2

    # Calculate the distance between the two points
    distance = np.linalg.norm(np.array(A) - np.array(point_with_highest_Y))
    print(f"Distance between A and the point with highest Y: {distance} pixels")
    print(f"Distance in cm: {distance/pixel_cm_ratio} cm")

else:
    print("No pixels found with the same X value as A.")

# Optionally display the result
cv2.namedWindow('Image with Line', cv2.WINDOW_NORMAL)
cv2.imshow('Image with Line', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

