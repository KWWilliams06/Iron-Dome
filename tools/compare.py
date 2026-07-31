import sys
import numpy as np
import cv2

a, b = sys.argv[1], sys.argv[2]

pa = cv2.imread(f"{a}_lb.png", cv2.IMREAD_COLOR).astype(np.int16)
pb = cv2.imread(f"{b}_lb.png", cv2.IMREAD_COLOR).astype(np.int16)
d = np.abs(pa - pb)
print(f"letterbox: shape {pa.shape} vs {pb.shape}, max diff {d.max()}, "
      f"pixels differing {np.count_nonzero(d)}")

ba = np.fromfile(f"{a}_blob.bin", dtype=np.float32)
bb = np.fromfile(f"{b}_blob.bin", dtype=np.float32)
print(f"blob: {ba.size} vs {bb.size} elems, max diff {np.abs(ba-bb).max()}")
