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
runtime.gpio.led(False, False)

# OpenCV 창 및 트랙바 생성 -> 현재 최초값은 최적화를 적용한 값
cv2.namedWindow('parameters')
cv2.createTrackbar('lower_S', 'parameters', 0, 255, lambda x: None)
cv2.createTrackbar('lower_V', 'parameters', 180, 255, lambda x: None)
cv2.createTrackbar('upper_S', 'parameters', 28, 255, lambda x: None)
cv2.createTrackbar('thresh', 'parameters', 175, 255, lambda x: None)
cv2.createTrackbar('minLineLength', 'parameters', 30, 100, lambda x: None)
cv2.createTrackbar('maxLineGap', 'parameters', 30, 100, lambda x: None)

# Flask 서버를 백그라운드에서 실행
server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()

# 변수 초기화
#current_action = None

#yolo_thread = threading.Thread(target=vision.cnn.yolo_inference_loop, daemon=True)
#yolo_thread.start()

try :
    while True :
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        runtime.flask_server.current_frame = frame
        
        # 2번 과정: 프레임을 그레이스케일로 변환
        ls = cv2.getTrackbarPos('lower_S', 'parameters')
        lv = cv2.getTrackbarPos('lower_V', 'parameters')
        us = cv2.getTrackbarPos('upper_S', 'parameters')
        # 보통 고정하는 값은 고정하고, 트랙바로 조정하는 값만 변수 설정
        lower_white = np.array([0, ls, lv])
        upper_white = np.array([180, us, 255])
        
        gray = vision.cv_module.origin_to_gray(frame, lower_white, upper_white)

        # 3번 과정: 그레이스케일에서 canny 엣지 검출     
        threshold_val = cv2.getTrackbarPos('thresh', 'parameters')
        canny = vision.cv_module.gray_to_canny(gray, threshold=threshold_val)
        
        # 4번 과정: 허프 변환으로 직선 검출
        minLineLength = cv2.getTrackbarPos('minLineLength', 'parameters')
        maxLineGap = cv2.getTrackbarPos('maxLineGap', 'parameters')

        lines = vision.cv_module.edges_to_lines(canny, minLineLength=minLineLength, maxLineGap=maxLineGap)
        processed_frame = vision.cv_module.draw_lines(canny, lines) # 실제 주행에서는 사용 X, Flask 서버에 전달할 프레임
        runtime.flask_server.processed_frame = processed_frame
      
        center_x = vision.cv_module.get_center_from_lines(lines, 380)
        angle = vision.cv_module.get_motor_angle(center_x)
        runtime.gpio.motor(50, 1, 1)
        runtime.gpio.servo(angle)

        # 지금 해야할 일: YOLO 모델을 이용한 객체 인식 -> 회피 코드, 정지 코드 작성
        # 지금 해야할 일: angle을 PID제어 과정 추가
#        with action_lock:
#            current_action = shared_action
        """
        match current_action:
            case "stop":
                runtime.gpio.motor(0, 1, 1)
            case "avoid":
                print('회피')
            case "traffic_light":
                print('신호등 감지')
            case None:
                result_x = vision.cv_module.get_center_from_lines(lines, 380) # y = 380에서 중심 좌표를 구함
                center = vision.cv_module.get_motor_angle(result_x)
                runtime.gpio.motor(50, 1, 1) # 아직 후진코드 작성 X
                runtime.gpio.servo(center)
                time.sleep(0.1)
        """
        
except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.stop_all()
    runtime.camera.release_camera()
    cv2.destroyAllWindows()
