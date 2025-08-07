import cv2
import runtime
from runtime.config import action_lock, shared_action
import threading
import time
import vision
import numpy as np

# main을 위한 초기화
runtime.gpio.init()
runtime.camera.init(640,480,30)

# Flask 서버를 백그라운드에서 실행
server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()

runtime.gpio.servo(90)
time.sleep(1)

try :
    while True :
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        runtime.flask_server.current_frame = frame

        # 2번 과정: 프레임을 그레이스케일로 변환
        gray = vision.cv_module.origin_to_gray(frame, lower_white=np.array([0,0,180]), upper_white=np.array([180,28,255]))

        # 3번 과정: 그레이스케일에서 canny 엣지 검출     
        canny = vision.cv_module.gray_to_canny(gray, threshold=175)
        
        # 4번 과정: 허프 변환으로 직선 검출
        lines = vision.cv_module.edges_to_lines(canny, minLineLength=30, maxLineGap=30)
        processed_frame = vision.cv_module.draw_lines(canny, lines, color = (0,255,0), thickness=2) # 실제 주행에서는 사용 X, Flask 서버에 전달할 프레임
        if len(processed_frame.shape) == 2:  # 흑백이면
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        runtime.flask_server.processed_frame = canny

        # center_x = vision.cv_module.get_center_from_lines(lines, 380)
        # angle = vision.cv_module.get_motor_angle(center_x) 
        #print("angle: ", angle)
        #runtime.gpio.motor(30, 1, 1)
        #runtime.gpio.servo(angle)
        get_center_from_canny = vision.cv_module.get_center_from_canny(canny, y = 380)
        angle = vision.cv_module.get_motor_angle(get_center_from_canny, 640)
        print('center:', get_center_from_canny)
        print('angle: ', angle)
        runtime.gpio.motor(30, 1, 1)
        runtime.gpio.servo(angle)
        time.sleep(1)
        # 지금 해야할 일: YOLO 모델을 이용한 객체 인식 -> 회피 코드, 정지 코드 작성
        # 지금 해야할 일: angle을 PID제어 과정 추가

except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.stop_all()
    runtime.camera.release_camera()

