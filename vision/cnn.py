import cv2
from ultralytics import YOLO
import time # YOLO 모델 시간 측정
# from runtime.config import YOLO_label     # 안씀
model = YOLO("datasets/runs/detect/train/weights/best.pt") # 학습 시킨 모델을 load

def _detect_class_id(frame):
    try:
        # 안전장치 코드
        if frame is None:
            return [], 0.0  

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 메인 추론 코드
        start = time.time()
        result = model(frame, imgsz=416, conf=0.15, iou=0.5, verbose=False)[0]   # 200 -> 416 으로 변경
        # 신뢰도 0.2 -> 0.15 낮춤
        print(result.boxes.conf.tolist()) # conf 출력
        print(f"YOLO 결과 load 시간: {time.time() - start:.4f}초")

        if result.boxes is None or len(result.boxes) == 0:
            return [], 0.0

        detected_cls_ids = result.boxes.cls.int().tolist()

        # 면적 계산: 가장 큰 박스
        areas = []
        for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
            areas.append((x2 - x1) * (y2 - y1))
        max_area = max(areas) if areas else 0.0

        return detected_cls_ids, float(max_area)

    except Exception as e:
        print("❌ YOLO 예측 오류:", e)
        return [], 0.0

def yolo_inference_loop(frame):
        print("추론 중")
        detected_cls_ids, area = _detect_class_id(frame)
        if (4 in detected_cls_ids) or (2 in detected_cls_ids):
            print('stop, area: ', area) 
            return YOLO_label.stop   
        elif 0 in detected_cls_ids:
            print('car, area: ', area) 
            return YOLO_label.car
        elif 1 in detected_cls_ids:
            print('left, area: ', area) 
            return YOLO_label.left
        elif 3 in detected_cls_ids:
            print('right, area: ', area)
            return YOLO_label.right      
        else:
            print('None, area: ', area)
            return None