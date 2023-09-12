# Importing Dependencies
import cv2
import torch
import numpy as np
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Device Configuration
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# Loading yolov5 model
def load_model(model_path="best.pt"):
    """
    Load the model from the given path
    Args:
        model_path (str): Path to the model
    Returns:
        model: Loaded model
    """
    return attempt_load(model_path, device=device)

# Preprocessing Image
def preprocess_image(uploaded_file):
    """
    Preprocess the image for inference
    Args:
        uploaded_file (str): Path to the image
    Returns:
        image: PIL Image
        img: Preprocessed image tensor
    """
    image = Image.open(uploaded_file)
    image = image.resize((640, 640))
    img = np.array(image)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)
    return image, img

# Getting Predictions
def get_predictions(model, img, conf_thres, iou_thres):
    """
    Get predictions from the model
    Args:
        model: Loaded model
        img: Preprocessed image tensor
        conf_thres (float): Confidence threshold
        iou_thres (float): IOU threshold
    Returns:
        pred: Predictions
    """
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    pred = [x.detach().cpu().numpy() for x in pred]
    pred = [x.astype(int) for x in pred]
    return pred

# Drawing Bounding Boxes
def draw_bounding_boxes(image, pred):
    """
    Draw bounding boxes on the image
    Args:
        image: PIL Image
        pred: Predictions
    Returns:
        image: Resultant PIL Image
        len(boxes): Number of detected objects
    """
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