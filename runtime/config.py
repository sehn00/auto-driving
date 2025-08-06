from enum import Enum
import threading

class YOLO_label(Enum):
    right = 0
    left = 1
    traffic_light = 2
    stop = 3
    car = 4
    

action_lock = threading.Lock()
shared_action = None