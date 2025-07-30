import runtime 
import threading # Flask 서버를 백그라운드에서 실행하기 위한 쓰레드
import time

# main을 위한 초기화
runtime.gpio.init()
runtime.camera.init(640,480,30)
runtime.gpio.led(False, False)

# 변수 선언
current_status = "go"
current_angle = 90

server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()


try :
    while True :
        print("서버 확인 중")
        
        
        
except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.stop_all()
    runtime.camera.release_camera()