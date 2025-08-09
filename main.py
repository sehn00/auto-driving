import cv2
import runtime
#from runtime.config import action_lock, shared_action
import runtime.config as config  
import threading
import time
import vision
import numpy as np
import keyboard

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

def set_action(val):
    with config.action_lock:
        config.shared_action = val
    print(f"acation = {val}")



try :
    while True :
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        runtime.flask_server.current_frame = frame

        # 1-plus 과정: BEV -> 이것을 좀 조정하면 좋을듯
        bev = vision.cv_module.origin_to_bev(frame)
        
        # 2번 과정: 프레임을 그레이스케일로 변환
        gray = vision.cv_module.origin_to_gray(bev, lower_white=np.array([0,0,180]), upper_white=np.array([180,28,255]))

        # 3번 과정: 그레이스케일에서 canny 엣지 검출     
        canny = vision.cv_module.gray_to_canny(gray, threshold=175)
        
        # 4번 과정: 허프 변환으로 직선 검출 -> 지금은 허프만 라인을 제대로 활용하고 있지 못함
        lines = vision.cv_module.edges_to_lines(canny, minLineLength=30, maxLineGap=30)
        processed_frame = vision.cv_module.draw_lines(canny, lines, color = (0,255,0), thickness=2) # 실제 주행에서는 사용 X, Flask 서버에 전달할 프레임
        if len(processed_frame.shape) == 2:  # 흑백이면
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        runtime.flask_server.processed_frame = canny

        # 5번 과정: canny로부터 중심과 대응하는 각도를 얻음
        get_center_from_canny = vision.cv_module.get_center_from_canny(canny, y = 300)
        angle = vision.cv_module.get_motor_angle(get_center_from_canny, 640)
        print('angle: ', angle)
        
        # 6번 과정: YOLO_thread로 처리한 상태를 읽음
#        with config.action_lock:
#            action = config.shared_action   # YOLO 쓰레드가 쓴 최신값을 읽음
        # 현재는 가상 상태 -> 상태를 결정
        keyboard.on_press_key('a', lambda e: set_action("avoid_left"))
        keyboard.on_press_key('b', lambda e: set_action("avoid_right"))        
        keyboard.on_press_key('s', lambda e: set_action("stop"))
        keyboard.on_press_key('t', lambda e: set_action("traffic_light"))
        keyboard.on_press_key('l', lambda e: set_action("left"))
        keyboard.on_press_key('r', lambda e: set_action("right"))
        keyboard.on_press_key('q', lambda e: set_action(None))

        # 7번 과정: 5번 과정 + 6번 과정을 종합하여 차량의 동작 상태를 결정함
        if config.shared_action == None:
            runtime.gpio.motor(30, 1, 1)
            runtime.gpio.servo(angle)   
        elif config.shared_action == "avoid_left":
            pass
        elif config.shared_action == "avoid_right":
            pass
        elif config.shared_action == "stop":
            runtime.gpio.motor(0,1,1)
            runtime.gpio.servo(90)
            time.sleep(3)
        elif config.shared_action == "traffic_light_red":
            pass
        elif config.shared_action == "traffic_light_green":
            pass
        elif config.shared_action == "left": # left에서 일정한 크기 이상의 바운딩 박스를 발견하면 왼쪽으로 
            runtime.gpio.motor(30, 1, 1)
            runtime.gpio.servo(30)
            time.sleep(2)
        elif config.shared_action =="right": # right에서 일정한 크기 이상의 바운딩 박스를 발견하면 오른쪽으로
            runtime.gpio.motor(30,1,1)
            runtime.gpio.servo(150)
            time.sleep(2)
            

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