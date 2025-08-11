import cv2
import runtime
import threading
import time
import vision
import numpy as np
from collections import deque

# main을 위한 초기화
runtime.gpio.init()
runtime.gpio.servo(90)
time.sleep(3)
runtime.camera.init(640,480,30)

# Flask 서버를 백그라운드에서 실행
server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()
print("Flask 서버가 백그라운드에서 실행 중입니다.")

yolo_thread = threading.Thread(target=vision.cnn.yolo_inference_loop, daemon=True)
yolo_thread.start()
print("yolo가 백그라운드에서 실행 중입니다.")

# 변수 초기화
runtime.config.shared_action = None

try :
    while True :
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        runtime.flask_server.current_frame = frame

        # 1-plus 과정: BEV -> 단점이 더 큰 것 같아 일단은 배제
        #bev = vision.cv_module.origin_to_bev(frame)
        frame = frame[frame.shape[0]//2:, :] # 사진 조정

        # 2번 과정: 프레임을 그레이스케일로 변환
        gray = vision.cv_module.origin_to_gray(frame)

        # 3번 과정: 그레이스케일에서 canny 엣지 검출     
        canny = vision.cv_module.gray_to_canny(gray)
        runtime.flask_server.processed_frame = canny
        get_center_from_canny = vision.cv_module.get_center_from_canny(canny, y = 380)
        angle = vision.cv_module.get_motor_angle(get_center_from_canny, 640)    
        print('get_center_from_canny', get_center_from_canny)
        print('angle', angle)
        # 6번 과정: YOLO_thread로 처리한 상태를 읽음
        with runtime.config.action_lock:
            action = runtime.config.shared_action   # YOLO 쓰레드가 쓴 최신값을 읽음

        match action:
            case runtime.config.YOLO_label.car:
                print(car)
                pass
            case runtime.config.YOLO_label.stop:
                runtime.gpio.motor(0,1,1)
                time.sleep(0.1)
            case runtime.config.YOLO_label.left:
                runtime.gpio.motor(50, 1, 1)
                runtime.gpio.servo(30)
                time.sleep(3)
            case runtime.config.YOLO_label.right:
                runtime.gpio.motor(50, 1, 1)
                runtime.gpio.servo(150)
                time.sleep(3)
            case None:
                runtime.gpio.motor(70, 1, 1)
                runtime.gpio.servo(angle)
                time.sleep(0.1)

except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.servo(90)
    print('servo 90도 정렬')
    time.sleep(1)
    runtime.gpio.stop_all()
    runtime.camera.release_camera()