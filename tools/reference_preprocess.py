import sys
import numpy as np
import cv2
from ultralytics.data.augment import LetterBox

src_path, prefix = sys.argv[1], sys.argv[2]

# cv2.imread, NOT PIL — different JPEG decoders disagree by +/-1 on chroma,
# which would show up as a preprocess bug that isn't one.
img = cv2.imread(src_path, cv2.IMREAD_COLOR)

# auto=False is the critical flag. The default (True) pads only to the next
# stride multiple, giving 640x384 for a 16:9 frame instead of 640x640.
lb = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)
out = lb(image=img)

cv2.imwrite(f"{prefix}_lb.png", out)

# Blob by hand — BGR->RGB, HWC->CHW, /255, add batch dim.
blob = out[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
blob.tofile(f"{prefix}_blob.bin")
