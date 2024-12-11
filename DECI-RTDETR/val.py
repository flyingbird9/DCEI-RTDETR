import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/rtdetr-r1822/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=1,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/train',
              name='runs/rtdetr-repvit_m23/val',
              )