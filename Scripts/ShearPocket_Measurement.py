# python3 ShearPocket_Measurement.py
# Shear Pocket dimension detection
import cv2
import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def process_image_pair(input_path1, input_path2, box_counter, previous_largest_x_coord=None, box_dimensions={}):
    parameters = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img1 = cv2.imread(input_path1)
    corners, _, _ = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=parameters)

    current_smallest_x_coord = None
    current_largest_x_coord = None

    if corners:
        aruco_perimeter = cv2.arcLength(corners[0], True)
        pixel_cm_ratio = aruco_perimeter / 600
        img2 = cv2.imread(input_path2)
        hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        l_b = np.array([0, 0, 138])
        u_b = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, l_b, u_b)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box_int = np.intp(box)
            cv2.drawContours(img2, [box_int], 0, (0, 255, 0), 2)

            x_coord = cv2.boundingRect(contour)[0]
            if current_smallest_x_coord is None or x_coord < current_smallest_x_coord:
                current_smallest_x_coord = x_coord
            if current_largest_x_coord is None or x_coord > current_largest_x_coord:
                current_largest_x_coord = x_coord

            if previous_largest_x_coord is not None and x_coord == current_largest_x_coord:
                assigned_number = box_counter - 1
            else:
                assigned_number = box_counter
                box_counter += 1

            top_left, top_right, bottom_right, bottom_left = box
            width = euclidean_distance(top_left, top_right) / pixel_cm_ratio
            height = euclidean_distance(top_left, bottom_left) / pixel_cm_ratio

            # Ensure height is always larger than width
            if width > height:
                width, height = height, width

            text_position = (int(top_left[0]), int(top_left[1]))
            cv2.putText(img2, str(assigned_number), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Update box dimensions dictionary
            if assigned_number in box_dimensions:
                box_dimensions[assigned_number].append((width, height))
            else:
                box_dimensions[assigned_number] = [(width, height)]

        cv2.namedWindow(f"Original Image: {input_path1}", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"Processed Image: {input_path1} and {input_path2}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Original Image: {input_path1}", img1)
        cv2.imshow(f"Processed Image: {input_path1} and {input_path2}", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return box_counter, current_largest_x_coord


if __name__ == "__main__":
    image_pairs = [
        ("../Images/Pocket1-Trans.jpg", "../Images/P1.jpg"),
        ("../Images/Pocket2-Trans.jpg", "../Images/P2.jpg"),

    ]

    image_pairs1 = [
        ("../Images/Pocket3-Trans.jpg", "../Images/P3.jpg"),
        ("../Images/Pocket4-Trans.jpg", "../Images/P4.jpg"),

    ]

    box_counter = 1
    previous_largest_x_coord = None

    # Dictionary to store dimensions for each assigned number
    box_dimensions = {}

    for input_path1, input_path2 in image_pairs:
        box_counter, previous_largest_x_coord = process_image_pair(input_path1, input_path2, box_counter, previous_largest_x_coord, box_dimensions)

    # Reset box_counter and previous_largest_x_coord
    box_counter = box_counter
    previous_largest_x_coord = None

    for input_path1, input_path2 in image_pairs1:
        box_counter, previous_largest_x_coord = process_image_pair(input_path1, input_path2, box_counter, previous_largest_x_coord, box_dimensions)

    # Calculate and display the average width and height for each assigned number
    for assigned_number, dimensions_list in box_dimensions.items():
        total_width = sum(width for width, _ in dimensions_list)
        total_height = sum(height for _, height in dimensions_list)
        average_width = total_width / len(dimensions_list)
        average_height = total_height / len(dimensions_list)
        print(f"Box {assigned_number}: Average Width = {average_width:.2f} cm, Average Height = {average_height:.2f} cm")
