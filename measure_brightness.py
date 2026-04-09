import cv2, numpy as np
from neural_swap import NeuralFaceSwapper, _align_face
sw = NeuralFaceSwapper(model_path="models/ghost_3_256.onnx", occlusion=False)
b = sw._backend
src = cv2.imread("source_faces/front_neutral.png")
sf = sw.detect(src)[0]
sa, _ = _align_face(src, sf.kps, b._size)
lab = cv2.cvtColor(sa, cv2.COLOR_BGR2Lab)
print(f"AI source L mean: {lab[:,:,0].mean():.1f}, std: {lab[:,:,0].std():.1f}")
r = cv2.imread("debug_aligned_compare.png")
w = r.shape[1] // 3
crop = r[:, w:2*w]
lab2 = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
print(f"Result     L mean: {lab2[:,:,0].mean():.1f}, std: {lab2[:,:,0].std():.1f}")
print(f"Gap: {lab[:,:,0].mean() - lab2[:,:,0].mean():.1f}")
