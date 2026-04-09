import cv2, numpy as np
r = cv2.imread("debug_aligned_compare.png")
h, w = r.shape[:2]
cw = w // 3
swapped = r[:, cw:2*cw]
source = r[:, 2*cw:]

def region_stats(img, y1f, y2f, x1f, x2f, label):
    ih, iw = img.shape[:2]
    region = img[int(ih*y1f):int(ih*y2f), int(iw*x1f):int(iw*x2f)]
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2Lab).astype(float)
    print(f"  {label}: L={lab[:,:,0].mean():.0f} A={lab[:,:,1].mean():.0f} B={lab[:,:,2].mean():.0f}")

print("Current result:")
region_stats(swapped, 0.5, 0.65, 0.15, 0.35, "Left cheek")
region_stats(swapped, 0.5, 0.65, 0.65, 0.85, "Right cheek")
region_stats(swapped, 0.15, 0.28, 0.3, 0.7, "Forehead")
region_stats(swapped, 0.30, 0.45, 0.25, 0.45, "Left eye area")
region_stats(swapped, 0.30, 0.45, 0.55, 0.75, "Right eye area")
region_stats(swapped, 0.45, 0.55, 0.40, 0.60, "Nose bridge")

print("\nAI source (TARGET):")
region_stats(source, 0.5, 0.65, 0.15, 0.35, "Left cheek")
region_stats(source, 0.5, 0.65, 0.65, 0.85, "Right cheek")
region_stats(source, 0.15, 0.28, 0.3, 0.7, "Forehead")
region_stats(source, 0.30, 0.45, 0.25, 0.45, "Left eye area")
region_stats(source, 0.30, 0.45, 0.55, 0.75, "Right eye area")
region_stats(source, 0.45, 0.55, 0.40, 0.60, "Nose bridge")
