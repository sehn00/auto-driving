from runtime import flask_server
import threading # Flask 서버를 백그라운드에서 실행하기 위한 쓰레드

server_thread = threading.Thread(target=flask_server.start_server, daemon=True)
server_thread.start()