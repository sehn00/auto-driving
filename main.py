import cv2
import runtime
import threading
import time
import vision
import numpy as np              # 추가
from collections import deque
from vision.pid_steer import pid_steer_from_center, pid

state = 0               # 0: 첫 번째 car 전, 1: 두 번째 car 전
last_car_time = 0       # 마지막 car 인식 시각 (초 단위 타임스탬프)

cooldown = 5.0          # 같은 장애물 반복 감지 방지 시간 (초)   <<< 실험적으로 수정해야 할 듯
turning_until = 0       # 회피(차선 변경) 유지 종료 시간 (초 단위 타임스탬프)
turning_angle = 90      # 현재 적용 중인 서보 각도 (90 = 직진)
lane_change_time = 2.0  # 차선 변경 유지 시간 (2초)

# 서보 각도 계산 주기(초) 추가
# CONTROL_INTERVAL = 0.1      # 10Hz

# main을 위한 초기화
runtime.gpio.init()
runtime.gpio.servo(90)
time.sleep(1)
runtime.camera.init(640, 480, 30)

# # Flask 서버를 백그라운드에서 실행
# server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
# server_thread.start()
# print("Flask 서버가 http://<라즈베리파이_IP>:5000 에서 실행 중입니다.")

# 추가
def _yolo_thread_func():
    while True:
        frame = runtime.camera.get_image()
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = vision.cnn.yolo_inference_loop(frame_rgb)  # 이 함수가 frame을 요구
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
last_control_time = 0   # 마지막 제어 시간 기록

try:
    start_time = time.monotonic()
    while True:
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        if frame is None:
            continue
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        now = time.monotonic()

        # 회피 상태(YOLO = car)가 켜져있으면, 어떤 라벨이 와도 
        if now < turning_until:    # 회피 유지 시간 동안은 고정 각/속도로 진행
            runtime.gpio.motor(20, 1, 1)    # 회피 중 속도 20
            runtime.gpio.servo(turning_angle)   # 각도: 좌회전은 30 / 우회전은 150
            continue
        # if current_time - last_control_time > CONTROL_INTERVAL:
        #     last_control_time = current_time
        # --- 아래 로직은 CONTROL_INTERVAL 마다 한 번씩만 실행됨 ---

        frame_1 = frame[frame.shape[0] // 2:, :] # 사진 조정 (하단 ROI)

        gray = vision.cv_module.origin_to_gray(frame_1)     # 그레이스케일로 변환
        canny = vision.cv_module.gray_to_canny(gray)    # 그레이스케일에서 canny 엣지 검출

        center_x = vision.cv_module.get_center_from_canny(canny)    # 중심 계산

        ## 차선 중심 스무딩
        hist.append(center_x)
        if len(hist) >= 4:
            vals = [v for v in hist if v is not None]
            center_smooth = sum(vals) / len(vals) if vals else None
        else:
            center_smooth = center_x

        ## PID 제어를 통해 서보 각도 계산
        servo_cmd = pid_steer_from_center(center_smooth, width=canny.shape[1])
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
                    # YOLO car 행동양식 시작 ============================
                    if now - last_car_time > cooldown:
                        last_car_time = now

                        if state == 0:      # 첫 car -> 왼쪽 차선으로 회피
                            turning_angle = 30
                            turning_until = now + lane_change_time  # 감지한 시간부터, lane_change_time (2s) 동안 회피 상태 유지
                            state = 1   # 다음엔 두 번째 car 처리
                            pid.reset()  # 회피 시작 전에 PID 상태 초기화 (급변 억제)
                            print("첫 번째 car: 왼쪽 회피 시작")

                        elif state == 1:    # 두 번째 car -> 오른쪽 차선으로 회피
                            turning_angle = 150
                            turning_until = now + lane_change_time
                            state = 2   # 이후에는 더 이상 트리거하지 않음(원하면 0으로 롤백)
                            pid.reset()
                            print("두 번째 car: 오른쪽 회피 시작")
                    # YOLO car 행동양식 끝 ============================

                case runtime.config.YOLO_label.stop:
                    print("YOLO 예측값: stop")
                    # ##### [ stop 객체 인식 후 1초동안 직진하게끔 ] #########################
                    # runtime.gpio.servo(90)
                    # runtime.gpio.motor(20, 1, 1)
                    # time.sleep(1)    # 1초간 저속 직진
                    
                    # runtime.gpio.motor(0, 1, 1)
                    # time.sleep(10)    # 10초간 정지
                    
                    continue    # 현재 루프의 나머지 부분 건너뛰기 (명령 충돌 방지용)

                case runtime.config.YOLO_label.left:
                    print("YOLO 예측값: left")
                    runtime.gpio.servo(30)
                    runtime.gpio.motor(20, 1, 1)
                    continue    # 현재 루프의 나머지 부분 건너뛰기 (명령 충돌 방지용)

                case runtime.config.YOLO_label.right:
                    print("YOLO 예측값: right")
                    runtime.gpio.servo(150)
                    runtime.gpio.motor(20, 1, 1)
                    continue    # 현재 루프의 나머지 부분 건너뛰기 (명령 충돌 방지용)
        # --------------------------------------------------

        # 5번 과정: YOLO 객체 미탐지 시 continue되지 않고, 아래의 기본 주행 로직이 실행됨
        ## 10초가 지나면 속도 35 -> 75
        # if time.time() - start_time <= 7:  # 이것도 새벽에 고쳤음!!!!! 까먹지말기
        if now - start_time <= 15:
            runtime.gpio.motor(35, 1, 1)
        elif now - start_time <= 20
		        runtime.gpio.motor(75, 1, 1)
        else:
            runtime.gpio.motor(35, 1, 1)

        runtime.gpio.servo(servo_cmd)

except KeyboardInterrupt:
    print("사용자 종료")

finally:
    runtime.gpio.servo(90)
    print('servo 90도 정렬')
    time.sleep(1)
    runtime.gpio.stop_all()
    runtime.camera.release_camera()