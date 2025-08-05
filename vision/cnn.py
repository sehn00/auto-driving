import cv2
from ultralytics import YOLO
import os
import time # YOLO 모델 시간 측정
import runtime
from runtime.config import YOLO_label, action_lock, shared_action

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")
model = YOLO(model_path) # 이 YOLO 모델을 학습 시킬 예정 main에서 vision을 import 시 자동으로 모델이 로드되게 설정


def detect_class_id(frame):
    try:
        if frame is None:
            print("❌ 프레임을 가져오지 못했습니다.")
            return []
        
        if frame.shape[2] == 4: # 만약 채널이 4개라면 RGB 형식으로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        start = time.time()
        result = model(frame, conf = 0.5)[0] # frame에 대한 예측 수행. 항상 list로 반환되기에 [0]으로 첫 번째 결과를 가져옴
        end = time.time()
        print(f"YOLO 결과 load 시간: {end - start:.4f}초") # YOLO 모델 로그 시간 측정. 불가피한 시간  
        detected_cls_ids = result.boxes.cls.astype(int).tolist() # 예측된 클래스 ID를 리스트로 변환

        return detected_cls_ids

    except Exception as e:
        print("❌ YOLO 예측 오류:", e)
        return []
    
def yolo_inference_loop():
    global shared_action
    while True:
        frame = runtime.camera.get_image()
        detected_cls_ids = detect_class_id(frame)
        action = None

        if YOLO_label.stop in detected_cls_ids:
            action = "stop"
        elif YOLO_label.car in detected_cls_ids:
            action = "avoid"
        elif YOLO_label.traffic_light in detected_cls_ids:
            action = "traffic_light"        
        elif YOLO_label.right in detected_cls_ids:
            action = "right"
        elif YOLO_label.left in detected_cls_ids:
            action = "left"
        else:
            action = None

        with action_lock:
            shared_action = action