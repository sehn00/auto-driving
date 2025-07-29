from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")



# 해당 method를 실행하여 서버를 시작합니다.
def start_server(): 
    app.run(host='0.0.0.0', port=5000, debug=False)

# push 요청...