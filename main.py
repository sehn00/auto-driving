import runtime 
import threading # Flask 서버를 백그라운드에서 실행하기 위한 쓰레드
import time
import vision
# main을 위한 초기화
runtime.gpio.init()
runtime.camera.init(640,480,5) # 일단 5FPS로 설정
runtime.gpio.led(False, False)

# 변수 선언
current_status = "go"
current_angle = 90

server_thread = threading.Thread(target=runtime.flask_server.start_server, daemon=True)
server_thread.start()


try :
    while True :
<<<<<<< HEAD
        start_time = time.time()
        frame = runtime.camera.get_image()
        runtime.flask_server.current_frame = frame
        end_time = time.time()
        print(f"⏱️ frame load 실행 시간: {end_time - start_time:.4f}초")
=======
        frame = runtime.camera.get_image() # 이미지를 받아옴
        runtime.flask_server.current_frame = frame # 해당 파일에 current_frame을 지금 사진으로 변경하면 웹서버에 반영됨

        roi_frame = vision.cv_module.ROI(frame)
        warp_frame = vision.cv_module.warp_image(roi_frame)
        pre_frame = vision.cv_module.pre_image(warp_frame)
        runtime.flask_server.processed_frame = pre_frame


        center, result_x = vision.cv_module.get_center(pre_frame)
        end_time = time.time()


        
>>>>>>> heo
        
        start_time = time.time()
        pre_frame = vision.cv_module.pre_image(frame)
        runtime.flask_server.processed_frame = pre_frame
        end_time = time.time()
        print(f"⏱️ frame process 실행 시간: {end_time - start_time:.4f}초")        
        
        start_time = time.time()
        center, result_x = vision.cv_module.get_center(pre_frame)
        end_time = time.time()
        print(f"{center:.4f}")
        print(f"⏱️ center 계산 실행 시간: {end_time - start_time:.4f}초")       

        # start_time = time.time() # 만약 center_x가 280이면 result_x = -40 이때, 좌회전을 해야함.
                

except KeyboardInterrupt: 
    print("사용자 종료")

finally:
    runtime.gpio.stop_all()
    runtime.camera.release_camera()