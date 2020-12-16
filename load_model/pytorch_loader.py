import sys
import torch
import numpy as np
import cv2
import PIL as Image

sys.path.append('models/')


def load_pytorch_model(model_path):
    model = torch.load(model_path)
    return model


def pytorch_inference(model, img_arr):
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    model.to(device)
    input_tensor = torch.tensor(img_arr).float().to(device)
    y_bboxes, y_scores, = model.forward(input_tensor)
    return y_bboxes.detach().cpu().numpy(), y_scores.detach().cpu().numpy()


def emotion_recognition(model, image):
    image = cv2.resize(image, (44, 44))

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, dtype=torch.float32)

    output = model(image)
    _, predicted = torch.max(output.data, dim=1)

    return predicted
