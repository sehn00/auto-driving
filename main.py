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
cooldown = 2.0          # 같은 장애물 반복 감지 방지 시간 (초)
turning_until = 0    # 회피(차선 변경) 유지 종료 시간 (초 단위 타임스탬프)
turning_angle = 90      # 현재 적용 중인 서보 각도 (90 = 직진)
lane_change_time = 2.0  # 차선 변경 유지 시간 (2초)

# 서보 각도 계산 주기(초) 추가
# CONTROL_INTERVAL = 0.1      # 10Hz

# main을 위한 초기화
runtime.gpio.init()
runtime.gpio.servo(90)
time.sleep(1)
runtime.camera.init(640, 480, 30)

# Flask 서버를 백그라운드에서 실행 (주석 해제)
server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()
print("Flask 서버가 http://<라즈베리파이_IP>:5000 에서 실행 중입니다.")

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
    start_time = time.time()
    while True:
        # 1번 과정: 카메라로부터 프레임을 가져옴
        frame = runtime.camera.get_image()
        if frame is None:
            continue
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # now = time.time()
        # if now - last_control_time > CONTROL_INTERVAL:
        #     last_control_time = now
        # --- 아래 로직은 CONTROL_INTERVAL 마다 한 번씩만 실행됨 ---

        frame_1 = frame[frame.shape[0] // 2:, :] # 사진 조정 (하단 ROI)

        # 2번 과정: 프레임을 그레이스케일로 변환
        gray = vision.cv_module.origin_to_gray(frame_1)

        # 3번 과정: 그레이스케일에서 canny 엣지 검출
        canny = vision.cv_module.gray_to_canny(gray)

        # (기존 주석 처리된 4번 과정)

        # 추가 ----------------------------------------------
        ## 중심 계산
        center_x = vision.cv_module.get_center_from_canny(canny)

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

        runtime.flask_server.current_frame = frame.copy()
        runtime.flask_server.processed_frame = canny.copy()

        # 원본 프레임 복사하여 디버깅 정보 그리기
        debug_frame = frame.copy()
        
        # Canny 엣지 이미지를 3채널 컬러로 변환 (합치기 위함)
        canny_color = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        
        # 하단 ROI에 해당하는 부분에만 선을 그림
        roi_h, roi_w = frame_1.shape[:2]
        
        # 화면 중앙선 (녹색)
        cv2.line(debug_frame, (320, 240), (320, 480), (0, 255, 0), 2)
        
        # 계산된 차선 중심선 (빨간색)
        if center_x is not None:
            # ROI 좌표계의 center_x를 전체 프레임 좌표계로 변환
            cv2.line(debug_frame, (center_x, 240), (center_x, 480), (0, 0, 255), 2)
        
        # 현재 서보 각도 텍스트로 표시
        cv2.putText(debug_frame, f"Servo CMD: {servo_cmd}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 상단: 원본+디버그, 하단: Canny 결과 (ROI 크기에 맞게 Canny 이미지 리사이즈)
        # canny_color의 높이를 240 (ROI 높이)으로 맞춤
        resized_canny = cv2.resize(canny_color, (640, 240))
        # 원본 프레임의 상단부와 합쳐서 전체 뷰 생성
        combined_view = np.vstack((debug_frame[:240], resized_canny))



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
                    runtime.gpio.motor(0, 1, 1)
                    runtime.gpio.servo(90)

                case runtime.config.YOLO_label.left:
                    print("YOLO 예측값: left")
                    runtime.gpio.servo(45)
                    runtime.gpio.motor(20, 1, 1)

                case runtime.config.YOLO_label.right:
                    print("YOLO 예측값: right")
                    runtime.gpio.servo(135)
                    runtime.gpio.motor(20, 1, 1)
        # --------------------------------------------------

        # 5번 과정: YOLO = interrupt로 외부 처리 되지 않는다면, 그냥 실행
        ## 7초가 지나면 속도 30 -> 20
        if time.time() - start_time <= 7:
            runtime.gpio.motor(30, 1, 1)  # 수정
        else:
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