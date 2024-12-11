import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    #model.load('runs/rtdetr-repvit_m23/weights/best.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=224,
                epochs=150,
                batch=2,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='runs',
                name='rtdetr-r18_group',
                )