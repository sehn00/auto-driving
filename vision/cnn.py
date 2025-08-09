import cv2
from ultralytics import YOLO
import os
import time # YOLO 모델 시간 측정
import runtime
from runtime.config import YOLO_label
import runtime.config as config 

#current_dir = os.path.dirname(os.path.abspath(__file__))
#model_path = os.path.join(current_dir, "yolo11n.pt") # 일단 기존 모델을 사용
model = YOLO("yolo11n.pt") # 이 YOLO 모델을 학습 시킬 예정 main에서 vision을 import 시 자동으로 모델이 로드되게 설정
# vision/best.pt 로 파일을 저장할 것

def detect_class_id(frame):
    try:
        if frame is None:
            print("❌ 프레임을 가져오지 못했습니다.")
            return []

        # BGRA → BGR (채널이 4개면)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        start = time.time()
        result = model(frame, conf=0.5, verbose=False)[0]
        end = time.time()
        print(f"YOLO 결과 load 시간: {end - start:.4f}초")

        # ✅ 여기만 수정
        # 방법 1: 텐서를 바로 int 리스트로
        detected_cls_ids = result.boxes.cls.int().tolist()

        # 방법 2: 넘파이로 변환 후 astype (취향)
        # detected_cls_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()

        return detected_cls_ids

    except Exception as e:
        print("❌ YOLO 예측 오류:", e)
        return []

    
def yolo_inference_loop():
    global shared_action
    while True:
        frame = runtime.camera.get_image()
        
        print("추론 중")
        detected_cls_ids = detect_class_id(frame)
        action = None
        number = None
        if 11 in detected_cls_ids:
            action = "stop"
            print(action)
        elif 2 in detected_cls_ids:
            action = "avoid"
            print(action)
        elif 9 in detected_cls_ids:
            action = "traffic_light"
            print(action)        
        else:
            action = None
            print(action)
            
        with config.action_lock:
            config.shared_action = action  # "stop"/"avoid"/"traffic_light"/None 등

