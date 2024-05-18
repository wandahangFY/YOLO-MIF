import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'yolov8n.pt')
    model.val(data=r'ultralytics/datasets/FLIR_ADAS_16.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/FLIR_ADAS_16',
              name='FLIR_ADAS_16-yolov8n-no_pretrained',
              )