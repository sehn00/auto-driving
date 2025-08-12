from enum import Enum
import numpy as np
import time
import threading

# YOLO 액션 공유 상태
action_lock = threading.Lock()   # 액션 값 보호용 락
shared_action = None             # 최신 YOLO 액션 저장 (None | YOLO_label)
shared_action_until = 0.0        # 액션 유효 시간(만료 타임스탬프, 선택사항)

class YOLO_label(Enum):
    car = 0
    left = 1
    red = 2
    right = 3
    stop = 4

# 2025-08-11 수정 항목
# class PID: 
#     def __init__(self, kp= 0.5, ki=0.003, kd=0.3, out_lim=(-20, 20),
#                 ei_lim=(-1e3, 1e3), d_alpha=0.15):
#         self.kp, self.ki, self.kd = kp, ki, kd
#         self.out_min, self.out_max = out_lim
#         self.ei_min, self.ei_max = ei_lim
#         self.d_alpha = float(np.clip(d_alpha, 1e-6, 1.0))

#         self.ei = 0.0 # 적분항에 해당하는 오차 누적값 ∫e(t) dt
#         self.e_prev = 0.0 # 이전 e(t)
#         self.d_f = 0.0 # 필터링된 de(t)/dt
#         self.u_prev = 0.0 # 이전 u(t)
#         # u(t) = Kp * e(t) + Ki * ∫ e(t) dt + Kd * de(t)/dt

#     def reset(self):
#         self.ei = 0.0
#         self.e_prev = 0.0
#         self.d_f = 0.0
#         self.u_prev = 0.0

#     def step(self, e, dt):
#         if dt <= 0:
#             return self.u_prev

#         # 미분 + 저역통과 필터
#         d_raw = (e - self.e_prev) / dt
#         self.d_f = (1 - self.d_alpha) * self.d_f + self.d_alpha * d_raw

#         # 적분(클램프)
#         self.ei = np.clip(self.ei + e * dt, self.ei_min, self.ei_max)

#         # 원출력
#         u = self.kp * e + self.ki * self.ei + self.kd * self.d_f

#         # 포화
#         u = float(np.clip(u, self.out_min, self.out_max))

#         # 상태 업데이트
#         self.e_prev = e
#         self.u_prev = u
#         return u
    
# PID_KP = 0.9
# PID_KI = 0.0
# PID_KD = 0.12
# PID_OUT_LIM = (-20, 20)    # steer(deg) 제한
# PID_D_ALPHA = 0.15         # D항 LPF 계수

# DT_MIN = 1e-3 # dt 하한
# DT_MAX = 0.05 # dt 상한 (50ms)
# LOST_TIMEOUT_S = 0.6 # 미검출 지속 시 중앙 복귀 시작 시간
# SERVO_RATE_DEG_PER_STEP = 5 # 서보 명령 한 주기 당 최대 변화량 (deg)

# # 차선 중심 Cx의 EMA 스무딩 계수 (0~1). 클수록 최신값 가중 ↑
# CX_SMOOTH_ALPHA = 0.25

# pid = PID(PID_KP, PID_KI, PID_KD, out_lim=PID_OUT_LIM, d_alpha=PID_D_ALPHA)
# prev_time = time.monotonic()  # 첫 호출 기준시간
# Cx_smooth = None              # EMA 상태 (픽셀 초기 None)
# last_servo_cmd = 90           # 직진 기준 90으로 시작(초기값)
# last_seen_time = time.monotonic()  # 마지막 유효 Cx 관측 시각

# def _rate_limit(prev, target, limit):
#     if target > prev + limit:
#         return prev + limit
#     if target < prev - limit:
#         return prev - limit
#     return target

# def pid_steer_from_center(Cx, width=640):
#     """
#     Cx: 검출된 차선 중심 x (픽셀) 또는 None
#     width: 프레임 폭 (픽셀)
#     return: 서보 명령 각도 [0, 180] (정수)
#     """
#     global prev_time, Cx_smooth, last_servo_cmd, last_seen_time
    
#     with action_lock:
#         now = time.monotonic()
#         raw_dt = now - prev_time
#         prev_time = now
        
#     if raw_dt <= 0:
#         return last_servo_cmd
    
#     dt = float(np.clip(raw_dt, DT_MIN, DT_MAX))
    
#     if width <= 0:
#         return last_servo_cmd

#     # Cx 스무딩 (EMA). 최초엔 측정값을 바로 채택.
#     if Cx is not None:
#         last_seen_time = now 
#         if Cx_smooth is None:
#             Cx_smooth = float(Cx)
#         else:
#             Cx_smooth = (1 - CX_SMOOTH_ALPHA) * Cx_smooth + CX_SMOOTH_ALPHA * float(Cx)
#     else:
#         # 미검출시 마지막 서보 명령 유지 (혹은 별도 정책: 완만히 중앙 복귀 등)
#         if (now - last_seen_time) > LOST_TIMEOUT_S:
#             target_cmd = 90
#             last_servo_cmd = int(_rate_limit(last_servo_cmd, target_cmd, SERVO_RATE_DEG_PER_STEP))
#             return last_servo_cmd
#         return last_servo_cmd
            
#     # 에러: (화면중앙 - 차선중앙). [-1, 1] 정규화
#     img_center = 0.5 * width
#     e = (img_center - Cx_smooth) / (0.5 * width) # class PID method에서 사용할 e 항목
    
#     if raw_dt >= 0.2:
#         pid.reset()
        
#     # PID로 deg 계산 → 서보 기준 각도로 변환 (직진=90)
#     steer = pid.step(e, dt)  # [-20, 20] # 이것을 통해서 steer을 구함 steer = u 
#     target_cmd = int(np.clip(90 + steer, 0, 180))
#     servo_cmd = int(_rate_limit(last_servo_cmd, target_cmd, SERVO_RATE_DEG_PER_STEP)) # 한 번에 최대 5도 이상 이동하지 못하게
#     last_servo_cmd = servo_cmd # servo 업데이트
#     return servo_cmd