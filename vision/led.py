# led.py: 터널을 감지했을 때, LED를 켜는 모듈
import runtime

def led_tunnel(gray):
    if gray.mean() < 50: # 터널 감지
        runtime.gpio.led(True, True) # 양쪽 LED를 켬
    else : 
        # nothing
        
def led_direction():
    # 방향 LED를 켜는 로직을 추가
    if ## :
        runtime.gpio.led(True, False) # 왼쪽 LED 켬
    elif ## :
        runtime.gpio.led(False, True) # 오른쪽 LED 켬
        
        
        
        
🚦 예시: 신호등 색 검출 실전 워크플로우
YOLO로 신호등(traffic light) 객체를 박스 검출

박스 내부 이미지만 crop

crop 이미지를 HSV 변환

색상(H 값) 기준으로 빨강/노랑/초록 판별