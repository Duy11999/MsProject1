#  python3 segpredict.py
from fastsam import FastSAM, FastSAMPrompt
import torch
import cv2
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt

model = FastSAM('FastSAM-x.pt')
IMAGE_PATH = './images/a(1).jpg'


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
everything_results = model(
    IMAGE_PATH,
    device=DEVICE,
    retina_masks=True,
    imgsz=1024,
    conf=0.8,
    iou=0.9,
)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# # everything prompt
ann = prompt_process.everything_prompt()

# # bbox prompt
# # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
# bboxes default shape [[0,0,0,0]] -> [[x1,y1,x2,y2]]
# ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])
# ann = prompt_process.box_prompt(bboxes=[[200, 200, 300, 300], [500, 500, 600, 600]])

# # text prompt
# ann = prompt_process.text_prompt(text='a photo of a dog')

# # point prompt
# # points default [[0,0]] [[x1,y1],[x2,y2]]
# # point_label default [0] [1,0] 0:background, 1:foreground
# ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

# point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
ann = prompt_process.point_prompt(points=[[812-55, 707]], pointlabel=[1])
ann1 = prompt_process.point_prompt(points=[[811, 559]], pointlabel=[1])
ann2 = prompt_process.point_prompt(points=[[1617, 706]], pointlabel=[1])
ann3 = prompt_process.point_prompt(points=[[1620, 851]], pointlabel=[1])
ann4 = prompt_process.point_prompt(points=[[800, 853]], pointlabel=[1])
ann5 = prompt_process.point_prompt(points=[[1618, 558]], pointlabel=[1])
ann6 = prompt_process.point_prompt(points=[[812, 410]], pointlabel=[1])
ann8 = prompt_process.point_prompt(points=[[1617, 409]], pointlabel=[1])

Ann = ann + ann1 + ann2 + ann3 + ann4 + ann5 + ann6 + ann8

prompt_process.plot(
    annotations=Ann,
    output_path='./output/visualization4.png',
    mask_random_color=False,
    better_quality=True,
    retina=False,
    withContours=False,
)

# Get the binary image

sv.plot_images_grid(
    images=ann+ann1+ann4+ann6,
    grid_size=(1, 3),
    size=(16, 4)
)

#cv2.imwrite('./output/binary3.jpg',(Ann[0]) *255)
#plt.show()
