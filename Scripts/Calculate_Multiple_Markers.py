import cv2
from object_detector import *
import numpy as np
import imutils
from PIL import Image

#Load Object Detector
detector = HomogeneousBgDetector()

def resize_by_ratio(image, height1):
    height, width = image.shape[:2]
    ratio = height / width
    new_width = int(height1 / ratio)
    new_height = height1

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def detect_aruco_corners(img, aruco_type):
    # Define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    # Load the input image from disk and resize it
    print("[INFO] Loading image...")
    image = cv2.imread(img)
    #resized_ratio = 0.5  # Resize image to 50% of its original size
    image = resize_by_ratio(image, 1500)
    #image = imutils.resize(image, width=600)

    # Verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(aruco_type, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_type))
        return None

    # Load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    print("[INFO] Detecting '{}' tags...".format(aruco_type))
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    # Verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # Flatten the ArUco IDs list
        ids = ids.flatten()

        # Display all detected marker IDs
        print("Detected marker IDs:")
        for markerID in ids:
            print(markerID)

        # Choose the marker to use based on its ID
        chosen_id = int(input("Enter the ID of the marker to use: "))

        # Find the chosen marker index
        chosen_marker_index = -1
        for i, markerID in enumerate(ids):
            if markerID == chosen_id:
                chosen_marker_index = i
                break

        if chosen_marker_index != -1:
            # Retrieve the chosen marker's corners
            chosen_marker_corners = corners[chosen_marker_index][0]

            # Convert the corner points to the expected data type
            chosen_marker_corners = chosen_marker_corners.astype(int)

            # Reshape the corners to match the expected format
            chosen_marker_corners = chosen_marker_corners.reshape((-1, 1, 2))

            # Return the corners of the chosen marker
            return np.array(chosen_marker_corners)

    # If no corners are found or the chosen marker is not detected, return None
    return None

# Example usage
img = "../Images/a(1).jpg"

#img = resize_by_ratio(img, 800)

aruco_type = "DICT_5X5_50"

chosen_marker_corners = detect_aruco_corners(img, aruco_type)
if chosen_marker_corners is not None:
    # Convert chosen_marker_corners to a numpy array
    chosen_marker_corners = np.array(chosen_marker_corners)

    # Calculate the perimeter of the chosen marker
    aruco_perimeter = cv2.arcLength(chosen_marker_corners, True)

    # Calculate the pixel-to-millimeter ratio
    pixel_mm_ratio = aruco_perimeter / 400

    print(pixel_mm_ratio)
else:
    print("No marker detected or invalid marker ID chosen.")

img = cv2.imread("../Images/a(1).jpg")
# resize image to match with resized ratio
img = resize_by_ratio(img, 1500)
#Draw Polygon around marker
int_conners = np.intp(chosen_marker_corners)
cv2.polylines(img,int_conners,True, (110, 115, 188), 5)

contours = detector.detect_objects(img)
#cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
#cv2.imshow("image1",img)


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
        object_width = w / pixel_mm_ratio
        object_height = h / pixel_mm_ratio
        print(object_height)
        print(object_width)

        center_x = x + (w // 2)
        center_y = y + (h // 2)

        # Display Width and Height on the center of the box
        text = "Width: {:.2f}".format(object_width)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)
        text_x = center_x - (text_size[0] // 2)
        text_y = center_y
        cv2.putText(img, text, (text_x, text_y-35), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

        text = "Height: {:.2f}".format(object_height)
        text_size, _ = cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN, 1, 2)
        text_x = center_x - (text_size[0] // 2)
        text_y = center_y + text_size[1] + 5
        cv2.putText(img, text, (text_x, text_y-35), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

        # Draw red circle at the center of the box
        cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)
