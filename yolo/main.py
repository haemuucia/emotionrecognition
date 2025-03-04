from ultralytics import YOLO
from PIL import Image
import cv2
#import os
#print(os.path.exists(r"D:\semester4\computer_vision\project\yolo\best.pt"))
model = YOLO(r"D:\semester4\computer_vision\project\yolo\best.pt")
results = model.predict(source="0", show=True)