import runtime 
import threading # Flask 서버를 백그라운드에서 실행하기 위한 쓰레드
import time
import vision
import numpy as np

# main을 위한 초기화
runtime.gpio.init()
runtime.camera.init(640,480,30)
runtime.gpio.led(False, False)

# 변수 선언
current_status = "go"
current_angle = 90

vertices = np.array([[ # 마스킹할 영역을 정의
    (100, 480),        # 좌하
    (280, 300),               # 좌상
    (360, 300),               # 우상
    (540, 480)]], dtype=np.int32) # 우하

src_pts = np.array([[ # 마스킹할 영역을 정의
    (40, 440),        # 좌하
    (280, 300),               # 좌상
    (360, 300),               # 우상
    (600, 440)]], dtype=np.float32) # 우하

server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()

threading.Thread(target = vision.cnn.detect_class_id, daemon = True).start()

try :
    while True :
        frame = runtime.camera.get_image() # 이미지를 받아옴
        ## 해당 frame은 numpy.ndarray 타입
        runtime.flask_server.current_frame = frame # 해당 파일에 current_frame을 지금 사진으로 변경하면 웹서버에 반영됨

        gray = vision.cv_module.origin_to_gray(frame)
        vision.led.led_on(gray) # 터널 감지 함수


        pre_frame = vision.cv_module.pre_image(frame)
        runtime.flask_server.processed_frame = pre_frame

        detected_cls_ids = vision.cnn.detect_class_id(frame)

        # LED 코드 





        result_x = vision.cv_module.get_center(pre_frame)
        if result_x == None:
            runtime.gpio.motor(-50)
            runtime.gpio.servo(90)
            time.sleep(0.4)
            continue
        elif result_x < -90:
            result_x = -90
        elif result_x >= 90:
            result_x = 90
        else: 
            runtime.gpio.motor(20)
            runtime.gpio.servo(90+int(result_x))
        print(result_x)

        
        
        
        
        
        
        
        
        
        
        
        
except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.stop_all()
    runtime.camera.release_camera()