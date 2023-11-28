# python3 perspectiveTrans.py
import numpy as np
import cv2
import math

def process_image(image_path, output_path):
    # Read input image and get corners of aruco
    img = cv2.imread(image_path)
    parameters = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    if len(corners) == 0:
        print(f"No markers found in {image_path}. Skipping.")
        return

    hh, ww = img.shape[:2]

    # Specify input coordinates for corners of detected marker
    input = np.float32(corners[0][0])

    # Output dimensions and coordinates
    width, height = 150, 150
    output = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Compute perspective matrix
    matrix = cv2.getPerspectiveTransform(input, output)

    # Calculate transformed coordinates and bounding box
    original_corners = np.float32([[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]])
    transformed_corners = cv2.perspectiveTransform(np.array([original_corners]), matrix)[0]
    min_x, max_x = np.min(transformed_corners[:, 0]), np.max(transformed_corners[:, 0])
    min_y, max_y = np.min(transformed_corners[:, 1]), np.max(transformed_corners[:, 1])
    bbox_width, bbox_height = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))

    # Translation matrix and final transformation
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    final_matrix = translation_matrix.dot(matrix)

    # Apply transformation and save result
    output_size = (bbox_width, bbox_height)
    imgOutput = cv2.warpPerspective(img, final_matrix, output_size, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Display the result
    cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Output Image", imgOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    cv2.imwrite(output_path, imgOutput)


# List of image paths
image_paths = ['../Images/a1.jpg', '../Images/a2.jpg', '../Images/20231124_135153.jpg', '../Images/a3.jpg']
output_paths = ['../Images/Pocket1-Trans.jpg', '../Images/Pocket2-Trans.jpg',  '../Images/Pocket3-Trans.jpg',  '../Images/Pocket4-Trans.jpg']

# Process each image
for img_path, out_path in zip(image_paths, output_paths):
    process_image(img_path, out_path)

