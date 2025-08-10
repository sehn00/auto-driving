#from roboflow import roboflow
import zipfile
import yaml
from ultralytics import YOLO

# ultralytics.checks()

#with zipfile.ZipFile("C:/Users/HeoYun/Desktop/auto_driving/object_detection.v1i.yolov11.zip") as target_file:
#    target_file.extractall('C:/Users/HeoYun/Desktop/auto_driving/')
'''

data = {'train' : 'train\images',
        'val' : 'valid/images',
        'test' : 'test/images',
      'names': ['car', 'left', 'red',
                'right', 'stop'],
        'nc' : 5}

with open('data.yaml', 'w') as f :
  yaml.dump(data, f)

with open('data.yaml', 'r') as f :
    my_yaml = yaml.safe_load(f)
    print(my_yaml)
'''

model = YOLO('yolo11n.pt')

model.train(data='data.yaml', epochs = 30, batch = 32, imgsz = 640, save = True, save_period = -1, 
            lr0 = 0.01, lrf = 0.1, augment = True, mixup = 0.2, device = "cpu", verbose = True,
            close_mosaic = 10, seed = 0, workers = 1, optimizer = 'AdamW' )