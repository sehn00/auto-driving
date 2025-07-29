from enum import Enum

stop_count = 0
stop_threshold = 3
brightness_threshold = 110

class Status(Enum) :
    go = 0
    left = 1
    right = 2
    back = 3
    stop = 4
    avoid = 5
    accelerate = 6
    decelerate = 7

class YOLO_label(Enum): # 크게 보면 go, back, stop 
    none = -1   # 추가
    left = 0
    straight = 1
    right = 2
    hill_up = 3
    hill_down = 4
    sign_left = 5
    sign_right = 6
    sign_tunnel= 7
    sign_stop = 8 # 차단기
    red_light = 9
    yellow_light = 10
    green_light = 11
    car = 12 # 정적 장애물 
# 차량 운행 알고리즘에 따라 label 변경 가능

PRIORITY_RULES= [
        Status.stop,
        Status.avoid,
        # Status.back, # back은 실제로 구현 x 
        Status.accelerate,
        Status.decelerate,
        Status.go,
        Status.left,
        Status.right
    ]

PRIORITY = [
    YOLO_label.sign_stop,
    YOLO_label.red_light,
    YOLO_label.car,

    YOLO_label.yellow_light,
    YOLO_label.green_light,

    YOLO_label.sign_tunnel,
    YOLO_label.sign_left,
    YOLO_label.sign_right,

    YOLO_label.hill_up,
    YOLO_label.hill_down,

    YOLO_label.left,
    YOLO_label.straight,
    YOLO_label.right   
]