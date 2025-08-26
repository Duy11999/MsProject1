import cv2
import numpy as np
from PIL import Image

# Load 3 binary images (grayscale)
img1 = cv2.imread("/home/d8/Work/Segment-Anything-Jyp/61.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/d8/Work/Segment-Anything-Jyp/62.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("/home/d8/Work/Segment-Anything-Jyp/63.jpg", cv2.IMREAD_GRAYSCALE)

# Ensure same size
assert img1.shape == img2.shape == img3.shape, "Images must be same size"

# Merge using bitwise OR
merged = cv2.bitwise_or(img1, img2)
merged = cv2.bitwise_or(merged, img3)

# Optionally threshold (to ensure binary 0/255)
_, merged_bin = cv2.threshold(merged, 127, 255, cv2.THRESH_BINARY)

# Save
cv2.imwrite("/home/d8/Work/Segment-Anything-Jyp/6merged.png", merged_bin)

# --- Show with cv2.imshow ---
cv2.namedWindow("Merged Binary", cv2.WINDOW_NORMAL)
cv2.imshow("Merged Binary", merged_bin)
cv2.waitKey(0)   # wait for key press
cv2.destroyAllWindows()

# --- Or show with PIL if running in notebook ---
Image.fromarray(merged_bin).show()
