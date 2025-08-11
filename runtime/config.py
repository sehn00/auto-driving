from enum import Enum
import threading
import time
from collections import deque
import numpy as np

action_lock = threading.Lock()
shared_action = None

class YOLO_label(Enum):
    car = 0
    left = 1
    red = 2
    right = 3
    stop = 4

# 2025-08-11 수정 항목
class PID: 
    def __init__(self, kp, ki, kd, out_lim=(-20, 20),
                ei_lim=(-1e3, 1e3), d_alpha=0.15):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_lim
        self.ei_min, self.ei_max = ei_lim
        self.d_alpha = float(np.clip(d_alpha, 1e-6, 1.0))

        self.ei = 0.0
        self.e_prev = 0.0
        self.d_f = 0.0
        self.u_prev = 0.0

    def reset(self):
        self.ei = 0.0
        self.e_prev = 0.0
        self.d_f = 0.0
        self.u_prev = 0.0

    def step(self, e, dt):
        if dt <= 0:
            return self.u_prev

        # 미분 + 저역통과 필터
        d_raw = (e - self.e_prev) / dt
        self.d_f = (1 - self.d_alpha) * self.d_f + self.d_alpha * d_raw

        # 적분(클램프)
        self.ei = np.clip(self.ei + e * dt, self.ei_min, self.ei_max)

        # 원출력
        u = self.kp * e + self.ki * self.ei + self.kd * self.d_f

        # 포화
        u = float(np.clip(u, self.out_min, self.out_max))

        # 상태 업데이트
        self.e_prev = e
        self.u_prev = u
        return u

def pid_steer_from_center(Cx, width = 640):
    """Cx: 검출된 차선 중심 x(픽셀). width: 프레임 폭."""
    global prev_time
    now = time.monotonic()
    dt = now - prev_time
    prev_time = now

    # 차선 미검출 시: 이전 조향 유지(또는 감속/정지 정책 선택)
    if Cx is None or width <= 0:
        return pid.u_prev


    # 오차: (영상중앙 - 차선중앙).  [-1,1]로 정규화하면 튠이 쉬움
    img_center = width * 0.5
    e = (img_center - Cx_smooth) / (width * 0.5)

    # PID로 조향각(deg) 생성
    steer = pid.step(e, dt)  # [-20, 20] 범위

    # (서보가 90이 직진인 경우)
    servo_cmd = int(np.clip(90 + steer, 0, 180))
    return servo_cmd