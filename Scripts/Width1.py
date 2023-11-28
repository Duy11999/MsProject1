# python3 Width1.py
#Width Measurement
import cv2
import numpy as np

# Constants and Parameters
ARUCO_DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11
DETECTOR_PARAMETERS = cv2.aruco.DetectorParameters()

class ImageProcessor:
    def __init__(self, aruco_dict_type, detector_params):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.parameters = detector_params

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def detect_markers(self, img):
        return cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.parameters)

    def find_contours(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([0, 0, 1])
        u_b = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, l_b, u_b)
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def euclidean_distance(self, point1, point2):
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def return_points(self, image_path, aruco_dict_type, detector_params, pixel_cm_ratio):
        # Load the image
        img = cv2.imread(image_path)
        # img = cv2.rotate(img, cv2.ROTATE_180) #<<<<<<<<<<<<<<<<<<<<<<
        # Convert the image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the HSV range and create a mask
        l_b = np.array([0, 0, 1])
        u_b = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, l_b, u_b)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # Get the two largest contours

        points = {}
        dimensions = {}
        for i, contour in enumerate(contours):
            # Get the bounding rectangle for the current contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # Get four vertices of the rotated rectangle
            box = np.intp(box)  # Convert vertices to integers
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            # Sort the box points for consistency
            box = sorted(box, key=lambda k: [k[0], k[1]])
            top_left, top_right, bottom_right, bottom_left = box

            # Assign points based on the box index
            if i == 0:
                # First box
                points['A'], points['B'] = top_left, top_right
                points['C'], points['D'] = bottom_right, bottom_left  # Top left and top right
            # elif i == 1:  # Second box
            # points['C'], points['D'] = bottom_right, bottom_left  # Bottom right and bottom left

        return img, points, dimensions

    def process_images(self, image_paths, should_rotate=False):
        processed_images = []
        for path in image_paths:
            img = cv2.imread(path)
            if should_rotate:
                img = cv2.rotate(img, cv2.ROTATE_180)
            processed_images.append(img)
        return processed_images
def main():
    # Constants and Parameters
    ARUCO_DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11
    DETECTOR_PARAMETERS = cv2.aruco.DetectorParameters()

    # Initialize the ImageProcessor
    processor = ImageProcessor(ARUCO_DICT_TYPE, DETECTOR_PARAMETERS)
    image_paths = ['../Images/Pocket22-Trans.jpg', '../Images/P22.jpg', '../Images/W11.jpg'] #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #First is original, Second is shape of shearpocket, Third is shape of slab
    img1, img2, img = processor.process_images(image_paths, should_rotate= False)

    # Process the first image
    #img1 = processor.load_image('../Images/Pocket22-Trans.jpg')
    # img1 = cv2.rotate(img1, cv2.ROTATE_180) ### for 2
    corners, _, _ = processor.detect_markers(img1)
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

    #img2 = processor.load_image('../Images/P22.jpg')
    #img2 = cv2.rotate(img2, cv2.ROTATE_180) ### for 2

    processed_img, points, dimensions = processor.return_points('../Images/P22.jpg', ARUCO_DICT_TYPE, DETECTOR_PARAMETERS, pixel_cm_ratio)

    # Extract the specific points
    A1, B, C, D = points['A'], points['B'], points['C'], points['D']
        # Print the coordinates of the points
    print("Coordinates of the points:")
    print("A (Top Left of Box 1):", A1)
    print("B (Top Right of Box 1):", B)
    print("C (Bottom Right of Box 2):", C)
    print("D (Bottom Left of Box 2):", D)

    # Print the dimensions of the boxes
    print("\nDimensions of the boxes (in pixels):")
    for i, dim in dimensions.items():
        print(f"Box {i + 1} - Width: {dim['width']:.2f} pixels, Height: {dim['height']:.2f} pixels")

    cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Processed Image', processed_img)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()

    # Read the second image
    #img = cv2.imread('../Images/W11.jpg')  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #img = cv2.rotate(img, cv2.ROTATE_180)

    # Find all back pixels in the image
    y_coords, x_coords = np.where(np.all(img == [255, 255, 255], axis=-1))  # [255,255,255] white color
    white_pixel_coords = list(zip(x_coords, y_coords))
    # Sort white pixels by y and then x
    sorted_white_pixel_coords = sorted(white_pixel_coords, key=lambda k: (k[1], k[0]))

    # Find the white point with the lowest X-value
    lowest_x_point = min(sorted_white_pixel_coords, key=lambda k: k[0])
    print(f"White point with the lowest X-value: {lowest_x_point}")
    cv2.circle(img1, lowest_x_point, radius=5, color=(0, 255, 0), thickness=-1)  # Circle in green color with thickness -1 (filled)

    A_x, A_y = A
    pixels_with_same_y_as_A = sorted([(x, y) for x, y in sorted_white_pixel_coords if y == A1[1]], key=lambda k: k[0])

    pixels_with_same_y_as_B = sorted([(x, y) for x, y in sorted_white_pixel_coords if y == B[1]], key=lambda k: k[0])

    pixels_with_same_y_as_C = sorted([(x, y) for x, y in sorted_white_pixel_coords if y == C[1]], key=lambda k: k[0])

    pixels_with_same_y_as_D = sorted([(x, y) for x, y in sorted_white_pixel_coords if y == D[1]], key=lambda k: k[0])

    if pixels_with_same_y_as_C:
        point_with_smallest_X_C = pixels_with_same_y_as_C[0]  # [0] is the first pixel after sorting by X
        print(f"Point with same Y as B and smallest X value of C: {point_with_smallest_X_C}")
    else:
        print("No pixels found with the same Y value as C.")

    if pixels_with_same_y_as_B:
        point_with_smallest_X_B = pixels_with_same_y_as_B[0]  # [0] is the first pixel after sorting by X
        print(f"Point with same Y as B and smallest X value of B: {point_with_smallest_X_B}")
    else:
        print("No pixels found with the same Y value as B.")

    # Print the point with the smallest X value among pixels with same Y as A
    if pixels_with_same_y_as_A:
        point_with_smallest_X_A = pixels_with_same_y_as_A[0]  # [0] is the first pixel after sorting by X
        print(f"Point with same Y as A and smallest X value: {point_with_smallest_X_A}")
    else:
        print("No pixels found with the same Y value as A.")

    # Print the point with the smallest X value among pixels with same Y as A
    if pixels_with_same_y_as_D:
        point_with_smallest_X_D = pixels_with_same_y_as_D[0]  # [0] is the first pixel after sorting by X
        print(f"Point with same Y as D and smallest X value: {point_with_smallest_X_D}")
    else:
        print("No pixels found with the same Y value as D.")

    # Fit a line using all points
    all_points = np.array(
        [point_with_smallest_X_D, point_with_smallest_X_C, point_with_smallest_X_B, point_with_smallest_X_A],
        dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

    ###### FIND THE INTERSECTION AND DRAW THE PENPERDICULAR LINE
    # Define the line length to be drawn, can cover the image size
    line_length = 1000

    # Calculate the starting and ending points of the line
    lefty = int((-x0 * vy / vx) + y0)
    righty = int(((img1.shape[1] - x0) * vy / vx) + y0)

    # Draw the line
    cv2.line(img1, (img1.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)

    # Calculate the slope of the best fit line
    slope_vy_vx = vy / vx

    # The slope of the perpendicular line
    perpendicular_slope = -1 / slope_vy_vx

    # Calculate the intercept of the perpendicular line using point A
    perpendicular_intercept = A[1] - (perpendicular_slope * A[0])
    # Solve for x first (x_intersect)
    x_intersect = (perpendicular_intercept - (y0 - (slope_vy_vx * x0))) / (slope_vy_vx - perpendicular_slope)

    # Then solve for y using the equation of the best fit line (y_intersect)
    y_intersect = (slope_vy_vx * x_intersect) + (y0 - (slope_vy_vx * x0))

    # Intersection point
    intersection_point = (int(x_intersect), int(y_intersect))

    # Draw the perpendicular line in red color from point A to the intersection point
    cv2.line(img1, A, intersection_point, (0, 0, 255), 2)

    # DISTANCE MEASUREMENT
    # Calculate the Euclidean distance in pixels
    distance_pixels = np.sqrt((intersection_point[0] - A[0]) ** 2 + (intersection_point[1] - A[1]) ** 2)

    # Convert the distance from pixels to centimeters using the pixel_cm_ratio
    distance_cm = distance_pixels * pixel_cm_ratio

    # Print the distance
    print(f"Distance between A and the intersection point in pixels: {distance_pixels}")
    print(f"Distance between A and the intersection point in centimeters: {distance_cm}")
    # Optionally, display the result
    cv2.namedWindow('Image with Best Fit Line', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Best Fit Line', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()