import time
import numpy as np
from runtime.config import action_lock  # PID에서 dt 계산 구간 보호용

class PID:
    """
    표준 PID + (D항 저역통과, 적분 클램프, 출력 포화)
    e: [-1, 1] 정규화된 에러
    step() 출력 u: '조향 추가량(deg)', 직진 90도 기준
    """
    def __init__(self, kp, ki, kd, out_lim=(-20, 20),
                 ei_lim=(-1e3, 1e3), d_alpha=0.15):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_lim
        self.ei_min, self.ei_max = ei_lim
        self.d_alpha = float(np.clip(d_alpha, 1e-6, 1.0))  # D항 LPF 계수

        self.ei = 0.0
        self.e_prev = 0.0
        self.d_f = 0.0
        self.u_prev = 0.0

    def reset(self):
        """상태 초기화 (액션 후 재진입 시 사용)."""
        self.ei = 0.0
        self.e_prev = 0.0
        self.d_f = 0.0
        self.u_prev = 0.0

    def step(self, e, dt):
        if dt <= 0:
            return self.u_prev

        # D항: 저역통과 필터 적용
        d_raw = (e - self.e_prev) / dt
        self.d_f = (1 - self.d_alpha) * self.d_f + self.d_alpha * d_raw

        # I항: 적분 클램프
        self.ei = np.clip(self.ei + e * dt, self.ei_min, self.ei_max)

        # PID 합산
        u = self.kp * e + self.ki * self.ei + self.kd * self.d_f

        # 출력 포화
        u = float(np.clip(u, self.out_min, self.out_max))

        self.e_prev = e
        self.u_prev = u
        return u


# ===== 파라미터 =====
PID_KP = 100        # 오차 크기에 대한 반응 강도 조절(조향 각도 결정), 초기값 0.9
PID_KI = 0.0       # 0.0
PID_KD = 3.0       # 반응 속도 조절, 클 수록 반응속도 빠름, 초기값 0.12
PID_OUT_LIM = (-120, 120)     # (-25, 25)
PID_D_ALPHA = 0.15

DT_MIN = 1e-3
DT_MAX = 0.05
LOST_TIMEOUT_S = 0.6
SERVO_RATE_DEG_PER_STEP = 5

CX_SMOOTH_ALPHA = 0.25

# ===== 상태 =====
pid = PID(PID_KP, PID_KI, PID_KD, out_lim=PID_OUT_LIM, d_alpha=PID_D_ALPHA)
prev_time = time.monotonic()
Cx_smooth = None
last_servo_cmd = 90
last_seen_time = time.monotonic()


def _rate_limit(prev, target, limit):
    """서보 변화율 제한."""
    if target > prev + limit:
        return prev + limit
    if target < prev - limit:
        return prev - limit
    return target


def pid_steer_from_center(Cx, width=640):
    """
    Cx: 차선 중심 x (픽셀) 또는 None
    width: 이미지 폭 (픽셀)
    return: 서보 명령 [0, 180]
    """
    global prev_time, Cx_smooth, last_servo_cmd, last_seen_time

    with action_lock:
        now = time.monotonic()
        raw_dt = now - prev_time
        prev_time = now

    if raw_dt <= 0:
        return last_servo_cmd

    dt = float(np.clip(raw_dt, DT_MIN, DT_MAX))

    if width <= 0:
        return last_servo_cmd

    # EMA 스무딩
    if Cx is not None:
        last_seen_time = now
        if Cx_smooth is None:
            Cx_smooth = float(Cx)
        else:
            Cx_smooth = (1 - CX_SMOOTH_ALPHA) * Cx_smooth + CX_SMOOTH_ALPHA * float(Cx)
    else:
        if (now - last_seen_time) > LOST_TIMEOUT_S:
            target_cmd = 90
            last_servo_cmd = int(_rate_limit(last_servo_cmd, target_cmd, SERVO_RATE_DEG_PER_STEP))
            return last_servo_cmd
        return last_servo_cmd

    e = (0.5 * width - Cx_smooth) / (0.5 * width)
    # print(f"e={e:.3f}, steer_pre_sat={pid.kp*e + pid.ki*pid.ei + pid.kd*pid.d_f:.2f}")

    if raw_dt >= 0.2:
        pid.reset()

    steer = pid.step(e, dt)
    target_cmd = int(np.clip(90 - steer, 0, 180))

    servo_cmd = int(_rate_limit(last_servo_cmd, target_cmd, SERVO_RATE_DEG_PER_STEP))
    last_servo_cmd = servo_cmd

    return servo_cmd
