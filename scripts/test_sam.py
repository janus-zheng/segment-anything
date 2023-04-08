import time

import numpy as np
import cv2
from imgcat import imgcat

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/home/jupyter/model/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image = cv2.imread('../436401b2-2596-42a8-8b55-01fcca60dad8.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)
input_point = np.array([[350, 350]])
input_label = np.array([1])

input_box = np.array([11, 161, 672, 376])

for _ in range(10):
    since = time.time()
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True,
    # )
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False,
    )
    print(f"time cost: {time.time()-since:.4f} seconds")

print(masks.shape, scores.shape)
top1_mask = masks[np.argmax(scores), :, :]
print(top1_mask.shape, np.max(top1_mask), np.min(top1_mask))
imgcat((top1_mask * 255).astype(np.uint8))