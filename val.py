import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'runs/train/LLVIP-R20/LLVIP-yolov8n-RGBT-midfusion-no_pre2/weights/best.pt')
    model.val(data=r'ultralytics/datasets/LLVIP_r20.yaml',
              split='val',
              imgsz=640,
              batch=16,
              channels=4,
              use_simotm='RGBT',
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/LLVIP_r20',
              name='LLVIP_r20-yolov8n-no_pretrained',
              )