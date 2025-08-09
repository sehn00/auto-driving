import cv2
import runtime
#from runtime.config import action_lock, shared_action
import runtime.config as config  
import threading
import time
import vision
import numpy as np
import sys, termios, tty, select
# main을 위한 초기화
runtime.gpio.init()
runtime.gpio.servo(90)
time.sleep(1)
runtime.camera.init(640,480,30)

# Flask 서버를 백그라운드에서 실행
server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()
print("Flask 서버가 백그라운드에서 실행 중입니다.")

#yolo_thread = threading.Thread(target=vision.cnn.yolo_inference_loop, daemon=True)
#yolo_thread.start()
print("yolo가 백그라운드에서 실행 중입니다.")
time.sleep(3)

# 사진 저장 
#save_dir = "captured_frames"
#os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

config.shared_action = None

# --- 터미널 모드 설정 (프로그램 시작 직후) ---
fd = sys.stdin.fileno()
old_term_attr = termios.tcgetattr(fd)
tty.setcbreak(fd)  # ✅ 엔터 없이 1글자씩 즉시 읽기

def get_key():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

try :
    while True :
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        runtime.flask_server.current_frame = frame

        # 1-plus 과정: BEV -> 이것을 좀 조정하면 좋을듯
        #bev = vision.cv_module.origin_to_bev(frame)
        frame = frame[frame.shape[0]//2:, :]

        # 2번 과정: 프레임을 그레이스케일로 변환
        gray = vision.cv_module.origin_to_gray(frame, lower_white=np.array([0,0,180]), upper_white=np.array([180,28,255]))

        # 3번 과정: 그레이스케일에서 canny 엣지 검출     
        canny = vision.cv_module.gray_to_canny(gray, threshold=175)
        
        # 4번 과정: 허프 변환으로 직선 검출 -> 지금은 허프만 라인을 제대로 활용하고 있지 못함
        lines = vision.cv_module.edges_to_lines(canny, minLineLength=30, maxLineGap=30)
        processed_frame = vision.cv_module.draw_lines(canny, lines, color = (0,255,0), thickness=2) # 실제 주행에서는 사용 X, Flask 서버에 전달할 프레임
        if len(processed_frame.shape) == 2:  # 흑백이면
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        runtime.flask_server.processed_frame = canny

        # 5번 과정: canny로부터 중심과 대응하는 각도를 얻음
        get_center_from_lines = vision.cv_module.get_center_from_lines(canny, y = 190)
        angle = vision.cv_module.get_motor_angle(get_center_from_lines, 640)
        print("center: ", get_center_from_lines)
        print('angle: ', angle)
        
        # 6번 과정: YOLO_thread로 처리한 상태를 읽음
#        with config.action_lock:
#            action = config.shared_action   # YOLO 쓰레드가 쓴 최신값을 읽음
        # 현재는 가상 상태 -> 상태를 결정
        # ✅ 키 1글자 즉시 읽기
        key = get_key()
        if key == 'a':
            print("avoid_left")
            # with config.action_lock: config.shared_action = "avoid_left"
        elif key == 'b':
            print("avoid_right")
            # with config.action_lock: config.shared_action = "avoid_right"
        elif key == 'l':
            print("left")
        elif key == 'r':
            print("right")
        elif key == 't':
            print("traffic_light")
        elif key == 's':
            print("stop")
        elif key == 'q':
            print("None")
            # with config.action_lock: config.shared_action = None

        # 제어 로직
        if key is None or key == 'q':
            runtime.gpio.motor(50, 1, 1)
            runtime.gpio.servo(angle)
        elif key == 's' or key == 't':
            runtime.gpio.motor(0, 1, 1)
            runtime.gpio.servo(90)
            time.sleep(3)
        elif key == 'l':
            runtime.gpio.motor(50, 1, 1)
            runtime.gpio.servo(20)
            time.sleep(5)
        elif key == 'r':
            runtime.gpio.motor(50, 1, 1)
            runtime.gpio.servo(160)
            time.sleep(5)
        # 'a','b','q'는 pass 그대로

        time.sleep(0.2)  # ✅ 과열 방지용

except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.servo(90)
    print('servo 90도 정렬')
    time.sleep(1)
    runtime.gpio.stop_all()
    runtime.camera.release_camera()

# YOLO 추론 사이즈 
# 회피 코드 
# PID 제어
# 정지 코드

"""
모자란 부분
이미지 전처리: 노이즈 제거


"""