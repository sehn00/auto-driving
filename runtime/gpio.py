# gpio.py

import lgpio    # Windows나 WSL에서는 지원되지않음, only RPi용  -> 경고밑줄 무시하기
import atexit
from ._gpio_pins import PINS # runtime/_gpio_pins.py에서 정의한 핀 번호들을 가져옴
# Create global lgpio instance
pi = None
atexit.register(lambda: stop_all()) # 종료 루틴 
# lgpio는 GPIO 핀을 제어하기 위한 라이브러리로, GPIO 핀을 직접 제어할 수 있게 해줌

try:
    pi = lgpio.gpiochip_open(0)
    print(f"[gpio] GPIO 자동 초기화 완료: 핸들 {pi}")
except Exception as e:
    print("❌ GPIO 초기화 실패:", e)
    pi = None

# Initialization
def init():
    # Motor pins
    global pi
    if pi is None:
        pi = lgpio.gpiochip_open(0)
        print("[gpio] GPIO 초기화 완료")
    for pin in [PINS.M1_IN1, PINS.M1_IN2, PINS.M1_PWM, # PWM : 속도 제어 IN1, IN2 : 방향 제어 STBY : 모터 활성화
                PINS.M2_IN1, PINS.M2_IN2, PINS.M2_PWM, PINS.STBY]:
        lgpio.gpio_claim_output(pi, pin, 0) # 핀을 출력으로 사용하겠다고 OS에 등록

    # Servo
    lgpio.gpio_claim_output(pi, PINS.SERVO_PIN, 0) # 서보 모터 핀 출력으로 설정

    # LEDs : 좌우 방향등용 LED 핀을 출력으로 설정
    lgpio.gpio_claim_output(pi, PINS.LED_RIGHT, 0)
    lgpio.gpio_claim_output(pi, PINS.LED_LEFT, 0)

    # Battery pins (optional input mode if needed)
    # for pin in [PINS.BAT_100, PINS.BAT_75, PINS.BAT_50, PINS.BAT_25]:
    for pin in [PINS.BAT_100, PINS.BAT_50, PINS.BAT_10]:
        lgpio.gpio_claim_input(pi, pin)

    # Motor enable : 모터 드라이브 핀을 출력으로 설정
    lgpio.gpio_write(pi, PINS.STBY, 1)


# Servo control (angle: 0 ~ 180)
def servo(angle = 90):
    pulse = 500 + (angle / 180.0) * 2000  # 500~2500us
    lgpio.tx_servo(pi, PINS.SERVO_PIN, int(pulse))


# Motor direction + speed
def motor(speed = 0, inverse = 1, motor_id = 1):
    """
    Control motor direction and speed.

    :param motor_id: 1 or 2 (M1 or M2 on TB6612FNG)
    :param speed: -255 ~ +255
    """
    if motor_id == 1:
        IN1, IN2, PWM = PINS.M1_IN1, PINS.M1_IN2, PINS.M1_PWM
    elif motor_id == 2:
        IN1, IN2, PWM = PINS.M2_IN1, PINS.M2_IN2, PINS.M2_PWM
    else:
        raise ValueError("motor_id must be 1 or 2")

    if speed * inverse > 0:
        lgpio.gpio_write(pi, IN1, 1)
        lgpio.gpio_write(pi, IN2, 0)
    elif speed * inverse < 0:
        lgpio.gpio_write(pi, IN1, 0)
        lgpio.gpio_write(pi, IN2, 1)
    else:
        lgpio.gpio_write(pi, IN1, 0)
        lgpio.gpio_write(pi, IN2, 0)

    duty = min(abs(speed), 255) / 255 * 100.0
    lgpio.tx_pwm(pi, PWM, 1000, duty)


# LED control
def led(left_on=False, right_on=False):
    lgpio.gpio_write(pi, PINS.LED_LEFT, 1 if left_on else 0)
    lgpio.gpio_write(pi, PINS.LED_RIGHT, 1 if right_on else 0)


# Clean shutdown
def stop_all():
    motor(0, 1)
    motor(0, 2)
    lgpio.tx_pwm(pi, PINS.SERVO_PIN, 50, 0)
    led(False, False)
    lgpio.gpio_write(pi, PINS.STBY, 0)

def stby(state=0):
    lgpio.gpio_write(pi, PINS.STBY, state)

# Battery level reading
def battery():
    """
    Count number of active battery GPIO inputs to determine level.
    3 pins high -> 100%
    2 pins high -> 50%
    1 pin high -> 10%
    0 pin high -> 0%
    """
    pins = [PINS.BAT_100, PINS.BAT_50, PINS.BAT_10]
    count = sum(lgpio.gpio_read(pi, pin) for pin in pins)

    if count == 3:
        return 100
    elif count == 2:
        return 50
    elif count == 1:
        return 10
    else:
        return 0