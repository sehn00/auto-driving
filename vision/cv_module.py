import cv2
import numpy as np

lower_white= np.array([0,0,200])
upper_white = np.array([180,50,255])

def warp_image(image, src_pts, dst_size=(640, 480)):
    width, height = dst_size
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, dst_size)
    return warped

def pre_image(frame) :
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return morphed
