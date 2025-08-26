#  python3 Image_Redirection.py
import cv2
import numpy as np  # Import NumPy


def detect_aruco_and_check_direction_from_image(img):
    # Check if the image is valid
    if img is None:
        print(f"Error: Invalid image provided")
        return []

    # Define ArUco parameters
    parameters = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # List to hold directions of detected markers
    directions = []

    # If markers are detected, draw them on the image
    if ids is not None:
        # Draw detected markers
        # cv2.aruco.drawDetectedMarkers(img, corners, ids)

        # Loop through detected markers
        for corner_set, marker_id in zip(corners, ids.flatten()):
            # Calculate the center of the marker
            center_x = int(np.mean(corner_set[0][:, 0]))  # Average x-coordinates of corners
            center_y = int(np.mean(corner_set[0][:, 1]))  # Average y-coordinates of corners

            # Get the first corner coordinates
            corner_1 = corner_set[0][0]  # Corner 1 is the first point

            # Determine the direction of Corner 1 relative to the center
            if corner_1[0] < center_x and corner_1[1] < center_y:
                direction = "Top Left"
            elif corner_1[0] > center_x and corner_1[1] < center_y:
                direction = "Top Right"
            elif corner_1[0] < center_x and corner_1[1] > center_y:
                direction = "Bottom Left"
            else:
                direction = "Bottom Right"

            # Append the direction to the list with marker ID
            directions.append((marker_id, direction))

    return directions  # Return the list of directions


def rotate_image(image, angle):
    """Rotate the image by the specified angle (in degrees)."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image  # No rotation for 0 degrees


# Example usage
image_paths = ["../Images/2.jpg", "../Images/4.jpg", "../Images/Length/3.jpg", "../Images/Length/4.jpg", "../Images/1.jpg", "../Images/4.jpg"]  # List of image paths
image_paths1 = ["../Images/22.jpg", "../Images/Middle/44.jpg", "../Images/Length/33.jpg", "../Images/Length/44.jpg", "../Images/11.jpg", "../Images/44.jpg" ]  # List of image paths

for img_path, img1_path in zip(image_paths, image_paths1):
    img = cv2.imread(img_path)
    directions = detect_aruco_and_check_direction_from_image(img)
    print(f"Directions from image {img_path}:", directions)

    img1 = cv2.imread(img1_path)
    initial_directions1 = detect_aruco_and_check_direction_from_image(img1)
    print(f"Directions from image {img1_path}:", initial_directions1)

    # If the directions don't match, rotate until they do
    angles = [0, 90, 180, 270]  # Define the rotation angles
    matched = False  # Flag to indicate if a match is found

    for angle in angles:
        img1 = cv2.imread(img1_path)  # Re-read the original image
        if img1 is None:
            print(f"Error: Could not read image at {img1_path}")
            break

        # Handle the case where angle is 0
        if angle == 0:
            # No need to rotate; use the original image
            directions1 = detect_aruco_and_check_direction_from_image(img1)
            print(f"Directions after rotating {angle} degrees:", directions1)
        else:
            # Rotate the image
            img1_rotated = rotate_image(img1, angle)

            # Check directions for the rotated image directly from memory
            directions1 = detect_aruco_and_check_direction_from_image(img1_rotated)
            print(f"Directions after rotating {angle} degrees:", directions1)

        # Compare directions and break the loop if they match
        if directions == directions1:
            img1_rotated = rotate_image(img1, angle)  # Re-assigning is unnecessary here; it's already done
            cv2.imwrite(img1_path, img1_rotated)  # Save the correctly rotated image to disk
            print(f"Directions matched after rotating {angle} degrees for image {img1_path}.")
            matched = True
            break

    if not matched:
        print(f"No matching directions found for image {img1_path} after all rotations.")
