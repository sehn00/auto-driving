import cv2
import runtime
import threading
import time
import vision
from collections import deque   # 추가
from vision.pid_steer import pid_steer_from_center, pid     # 추가

# main을 위한 초기화
runtime.gpio.init()
runtime.gpio.servo(90)
time.sleep(1)
runtime.camera.init(640,480,30)

# Flask 서버를 백그라운드에서 실행
#server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
#server_thread.start()
#print("Flask 서버가 백그라운드에서 실행 중입니다.")

# 추가
def _yolo_thread_func():
    while True:
        frame = runtime.camera.get_image()
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = vision.cnn.yolo_inference_loop(frame_rgb)  # ← 이 함수가 frame을 요구
        with runtime.config.action_lock:
            runtime.config.shared_action = pred

# YOLO 스레드 실행
yolo_thread = threading.Thread(target=_yolo_thread_func, daemon=True)
yolo_thread.start()
print("yolo가 백그라운드에서 실행 중입니다.")

# 스무딩 버퍼
hist = deque(maxlen=5)      # 추가

# 변수 초기화
last_angle = 90

try :
    sss = time.time()
    while True :
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        if frame is None:
            continue
        # runtime.flask_server.current_frame = frame
        frame_1 = frame[frame.shape[0]//2:, :] # 사진 조정 (하단 ROI)

        # 2번 과정: 프레임을 그레이스케일로 변환
        gray = vision.cv_module.origin_to_gray(frame_1)

        # 3번 과정: 그레이스케일에서 canny 엣지 검출     
        canny = vision.cv_module.gray_to_canny(gray)
        # runtime.flask_server.processed_frame = canny

        # # 4번 과정: canny로부터 중심과 대응하는 각도를 얻음
        # get_center_from_canny = vision.cv_module.get_center_from_canny(canny, y = 190)
        # angle = vision.cv_module.get_motor_angle(get_center_from_canny,last_angle)
        # last_angle = angle
        # print('angle', angle)

        # 추가 ----------------------------------------------
        ## 중심 계산
        center_x = vision.cv_module.get_center_from_canny(canny)
        print(f'center_x: {center_x}')

        ## 차선 중심 스무딩
        hist.append(center_x)
        if len(hist) >= 4:
            vals = [v for v in hist if v is not None]
            center_smooth = sum(vals)/len(vals) if vals else None
        else:
            center_smooth = center_x

        ## PID 제어를 통해 서보 각도 계산
        servo_cmd = pid_steer_from_center(center_smooth, width = canny.shape[1])
        print(f'서보 각도: {servo_cmd}')

        ## YOLO 예측 읽기
        with runtime.config.action_lock:
            pred = runtime.config.shared_action # 최근에 감지한 액션 결과
        
        if pred is None:
            print("YOLO 예측값: 없음")
        else:
            match pred:
                case runtime.config.YOLO_label.car:
                    print("YOLO 예측값: car")
                case runtime.config.YOLO_label.stop:
                    print("YOLO 예측값: stop")
                case runtime.config.YOLO_label.left:
                    print("YOLO 예측값: left")
                case runtime.config.YOLO_label.right:
                    print("YOLO 예측값: right")
        # --------------------------------------------------

        # 5번 과정: YOLO = interrupt로 외부 처리 되지 않는다면, 그냥 실행
        ## 7초가 지나면 속도 30 -> 20
        if time.time() - sss <= 7 :
            runtime.gpio.motor(40,1,1)   # 수정
        else :
            runtime.gpio.motor(30, 1, 1) # 수정

        runtime.gpio.servo(servo_cmd)
        

except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.servo(90)
    print('servo 90도 정렬')
    time.sleep(1)
    runtime.gpio.stop_all()
    runtime.camera.release_camera()