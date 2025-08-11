import cv2
from ultralytics import YOLO
import time # YOLO 모델 시간 측정
import runtime
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
        result = model(frame, imgsz= 320, conf=0.2,
                    iou= 0.5, verbose=False)[0]
        end = time.time()
        print(f"YOLO 결과 load 시간: {end - start:.4f}초")

        if result.boxes is None or len(result.boxes) == 0:
            return []
        detected_cls_ids = result.boxes.cls.int().tolist()
        
        return detected_cls_ids

    except Exception as e:
        print("❌ YOLO 예측 오류:", e)
        return []

def yolo_inference_loop():
    while True:
        frame = runtime.camera.get_image()
        print("추론 중")
        detected_cls_ids = None
        detected_cls_ids = _detect_class_id(frame)
        print('detected_cls_ids', detected_cls_ids)
        if 4 or 2 in detected_cls_ids:
            # action = runtime.config.YOLO_label.stop
            print('stop')       
        elif 0 in detected_cls_ids:
            action = runtime.config.YOLO_label.car
            print('car')      
        elif 1 in detected_cls_ids:
            action = runtime.config.YOLO_label.left
            print('left')      
        elif 3 in detected_cls_ids:
            action = runtime.config.YOLO_label.right     
            print('right')              
        else:
            action = None
            print('None area', area)      
            
        with runtime.config.action_lock:
            runtime.config.shared_action = action

