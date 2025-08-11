import cv2
from ultralytics import YOLO
import time # YOLO 모델 시간 측정
import runtime
from runtime.config import YOLO_label
import runtime.config as config 
import numpy as np
model = YOLO("datasets/runs/detect/train/weights/best.pt") # 학습 시킨 모델을 load

def _detect_class_id(frame):
    try:
        if frame is None:
            print("❌ 프레임을 가져오지 못했습니다.")
            return []

        # BGRA → BGR (채널이 4개면)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
        start = time.time()
        result = model(frame, imgsz= 320, conf=0.35,
                       iou= 0.5, verbose=False)[0]
        end = time.time()
        print(f"YOLO 결과 load 시간: {end - start:.4f}초")

        if result.boxes is None or len(result.boxes) == 0:
            return []
        detected_cls_ids = result.boxes.cls.int().tolist()
        area = (result.boxes.x2-result.boxes.x2-1)*(result.boxes.y2-result.boxes.y1)
        
        return detected_cls_ids, area

    except Exception as e:
        print("❌ YOLO 예측 오류:", e)
        return []

def yolo_inference_loop():
    global shared_action
    while True:
        frame = runtime.camera.get_image()
        print("추론 중")
        detected_cls_ids, area = _detect_class_id(frame)
        if 4 or 2 in detected_cls_ids:
            action = YOLO_label.stop
            print('area', area)       
        elif 0 in detected_cls_ids:
            action = YOLO_label.car
            print('area', area)      
        elif 1 in detected_cls_ids:
            action = YOLO_label.left
            print('area', area)      
        elif 3 in detected_cls_ids:
            action = YOLO_label.right     
            print('area', area)              
        else:
            action = None
            print('area', area)      
            
        with config.action_lock:
            config.shared_action = action

