import cv2
from ultralytics import YOLO
import os
import time # YOLO 모델 시간 측정
import runtime
from runtime.config import YOLO_label, action_lock, shared_action

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "yolo11n.pt") # 일단 기존 모델을 사용
model = YOLO(model_path) # 이 YOLO 모델을 학습 시킬 예정 main에서 vision을 import 시 자동으로 모델이 로드되게 설정
# vision/best.pt 로 파일을 저장할 것

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

        if 11 in detected_cls_ids:
            action = "stop"
        elif 2 in detected_cls_ids:
            action = "avoid"
        elif 9 in detected_cls_ids:
            action = "traffic_light"        
        else:
            action = None

        with action_lock:
            shared_action = action