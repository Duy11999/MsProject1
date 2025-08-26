import cv2
import numpy as np
from pathlib import Path

def canny_edges(gray, blur=3, t1=50, t2=200):
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (blur|1, blur|1), 0)
    edges = cv2.Canny(gray, t1, t2)
    return edges

def sobel_edges(gray, ksize=3):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    return mag

def main():
    # 👉 Just set your input image here
    img_path = "/home/d8/Work/Master_Project/Images/Emseoung/2/66.jpg"   # change this filename
    method = "canny"         # "canny" or "sobel"

    # Load image
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to read image (unsupported format?)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply method
    if method == "canny":
        edges = canny_edges(gray, blur=3, t1=100, t2=200)
    else:
        edges = sobel_edges(gray, ksize=3)

    # Show original + edges
    h = max(img.shape[0], edges.shape[0])
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def resize_to_h(im, h):
        scale = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1]*scale), h), interpolation=cv2.INTER_AREA)

    vis = np.hstack([resize_to_h(img, h), resize_to_h(edges_bgr, h)])
    cv2.namedWindow("Original | Edges", cv2.WINDOW_NORMAL)
    cv2.imshow("Original | Edges", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
