import runtime 
import threading # Flask 서버를 백그라운드에서 실행하기 위한 쓰레드
#import time
import vision
import numpy as np

# main을 위한 초기화
runtime.gpio.init()
runtime.camera.init(640,480,30)
runtime.gpio.led(False, False)

# Flask 서버를 백그라운드에서 실행
server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()

# threading.Thread(target = vision.cnn.detect_class_id, daemon = True).start()

try :
    while True :
        frame = runtime.camera.get_image() # 이미지를 받아옴
        runtime.flask_server.current_frame = frame # 해당 파일에 current_frame을 지금 사진으로 변경하면 웹서버에 반영됨

        gray = vision.cv_module.origin_to_gray(frame)
        canny = vision.cv_module.gray_to_canny(gray)
        lines = vision.cv_module.edges_to_lines(canny)
        pre_frame = vision.cv_module.draw_lines(canny, lines) # 실제 주행에서는 사용 X
        runtime.flask_server.processed_frame = pre_frame # 전치리된 이미지를 웹서버에 반영

#        detected_cls_ids = vision.cnn.detect_class_id(frame) # CNN을 통한 객체 인식
        
        result_x = vision.cv_module.get_center_from_lines(lines, 380) # y = 380에서 중심 좌표를 구함
        center = vision.cv_module.get_motor_angle(result_x)
        runtime.gpio.motor(50, 1, 1)
        runtime.gpio.servo(center)
except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.stop_all()
    runtime.camera.release_camera()