import cv2
import torch
import numpy as np
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

device = select_device('0' if torch.cuda.is_available() else 'cpu')

def load_model(model_path="best.pt"):
    return attempt_load(model_path, device=device)

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((640, 640))
    img = np.array(image)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)
    return image, img

def get_predictions(model, img, conf_thres, iou_thres):
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    pred = [x.detach().cpu().numpy() for x in pred]
    pred = [x.astype(int) for x in pred]
    return pred

def draw_bounding_boxes(image, pred):
    boxes = []
    
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = det[:, :4] / 640 * image.size[0]
            for *xyxy, conf, cls in det:
                boxes.append(xyxy)

    img_np = np.array(image)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    return Image.fromarray(img_np), len(boxes)