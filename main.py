# from runtime import flask_server
import threading # Flask 서버를 백그라운드에서 실행하기 위한 쓰레드
import test
import time

current_status = "go"
current_angle = 90

"""
server_thread = threading.Thread(target=flask_server.start_server, daemon=True)
server_thread.start()
"""
server_thread = threading.Thread(target=test.start_server, daemon = True)
server_thread.start()

while(1) :
    print('실행 중...')
    current_angle += 1
    test.update_state(current_status, current_angle)
    time.sleep(3)  