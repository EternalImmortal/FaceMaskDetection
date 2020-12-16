import torch
import cv2
from vgg import VGG
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("Video open failed.")

status = True
idx = 0

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)
model = torch.load('VGG19', map_location=device)

while status:
    idx += 1
    status, img_raw = cap.read()
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(img_raw, (44, 44))
    if idx < 20:
        Image.fromarray(image).show()

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)


    image = torch.tensor(image, dtype=torch.float32)



    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output.data, dim=1)
    print(output)
