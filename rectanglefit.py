import cv2
import numpy as np

EDGE_IMG_PATH   = r"/home/d8/Downloads/6/canny_inside_box_3.png" #12
TARGET_IMG_PATH = r"/home/d8/Work/Master_Project/Images/Emseoung/2/66.jpg"

edge_gray = cv2.imread(EDGE_IMG_PATH, cv2.IMREAD_GRAYSCALE)
target    = cv2.imread(TARGET_IMG_PATH, cv2.IMREAD_COLOR)

if edge_gray is None or target is None:
    raise FileNotFoundError("Check EDGE_IMG_PATH / TARGET_IMG_PATH")
if edge_gray.shape[:2] != target.shape[:2]:
    raise ValueError("Images must be the same HxW")

# 1) binary edges -> solid mask of largest contour
_, bin_img = cv2.threshold(edge_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
bin_img = cv2.dilate(bin_img, np.ones((3,3), np.uint8), iterations=1)
cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = [c for c in cnts if cv2.contourArea(c) > 500]
if not cnts:
    raise RuntimeError("No usable contours found")
largest = max(cnts, key=cv2.contourArea)

H, W = edge_gray.shape[:2]
mask = np.zeros((H, W), np.uint8)
cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

# --- Step A: Erode to create an "always-inside" region (safety margin) ---
INSIDE_MARGIN = 6  # pixels; increase for more clearance
ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2*INSIDE_MARGIN+1, 2*INSIDE_MARGIN+1))
safe_mask = cv2.erode(mask, ker, iterations=1)
if cv2.countNonZero(safe_mask) == 0:
    safe_mask = mask.copy()  # fall back if margin too big

# --- Fit rectangle on the safe region ---
cnts_in, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
inner_cnt = max(cnts_in, key=cv2.contourArea)
rect = cv2.minAreaRect(inner_cnt)   # already inside thanks to erosion

# --- Step B: Binary-search tighten to ensure every pixel is inside the ORIGINAL contour ---
def rect_pts(r):
    return cv2.boxPoints(r).astype(np.float32)

def scale_rect(r, s):
    (cx, cy), (w, h), a = r
    return ((cx, cy), (max(1.0, w*s), max(1.0, h*s)), a)

# raster check: rectangle pixels must be subset of original mask
def fits_inside(r, base_mask):
    poly = rect_pts(r).astype(np.int32)
    test = np.zeros_like(base_mask)
    cv2.fillPoly(test, [poly], 255)
    outside = cv2.countNonZero(cv2.bitwise_and(test, cv2.bitwise_not(base_mask)))
    return outside == 0

# search from 100% size downward (should already fit; this just guarantees it)
lo, hi = 0.70, 1.00
best = scale_rect(rect, lo)
for _ in range(25):
    mid = (lo + hi) / 2.0
    cand = scale_rect(rect, mid)
    if fits_inside(cand, mask):
        best = cand
        lo = mid
    else:
        hi = mid

box = np.int32(np.round(rect_pts(best)))

# --- draw: red contour, green guaranteed-inside rectangle ---
out = target.copy()
cv2.drawContours(out, [largest], -1, (0, 0, 255), 2)
cv2.polylines(out, [box], True, (0, 255, 0), 3)

print("Guaranteed-inside rectangle (center, size, angle):", best)
print("Corner points:\n", box)

cv2.namedWindow("Guaranteed inside rectangle", cv2.WINDOW_NORMAL)
cv2.imshow("Guaranteed inside rectangle", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
