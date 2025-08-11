# 자율주행 경진대회
- SW를 중심으로 차선인식 및 장애물 회피를 이용한 자율주행경진대회
# 프로젝트 파일 구성
## main.py
프로젝트 main 코드  
## runtime/
_gpio_pins.py: gpio pin을 정의하기 위한 내부 코드  
camera.py: 카메라 동작을 위한 코드  
config.py: 각종 클래스와 딕셔너리 등등을 저장하기 위한 코드  
flask_server.py: flask 웹 서버를 개설을 위한 코드  
gpio.py: gpio 동작을 위한 함수를 구현한 코드  
## vision/
cnn.py: YOLO를 이용한 함수를 구현한 코드  
cv_module.py: 컴퓨터 비전을 이용한 함수를 구현한 코드  
## static/  
CSS, JS, 이미지, 글꼴 등 정적 파일들을 저장하는 폴더  
## templates/  
HTML 파일들을 저장하는 폴더  
## requirmentes.txt
해당 프로젝트를 진행하면서 다운로드 받을 모듈을 정리한 텍스트 파일  
## datasets/
YOLO 학습을 위한 이미지와 학습 결과 model 저장  
## cap/
라즈베리파이 카메라를 통해 캡처한 이미지를 저장한 폴더  
## test/
각종 함수나 코드의 test를 위한 임시 폴더  