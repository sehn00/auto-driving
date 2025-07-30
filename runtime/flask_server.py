from flask import Flask, render_template, Response
from runtime import camera
import cv2
app = Flask(__name__)

# 전역 변수 초기화
current_status = "init"
current_angle = 90
current_frame = None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/status')
def status():
    return {
        'status': str(current_status),
        'angle': current_angle
    }
    
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')    

def generate():
    global current_frame
    while True:
        if current_frame is None:
            continue
        ret, jpeg = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')    

# 외부에서 상태 업데이트 가능하게
def update_state(status, angle):
    global current_status, current_angle
    current_status = status
    current_angle = angle

# 해당 method를 실행하여 서버를 시작합니다.
def start_server(): 
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)