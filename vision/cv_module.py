import cv2
import numpy as np

#lower_white= np.array([0,0,120])
#upper_white = np.array([180,50,255])

def origin_to_gray(frame, lower_white=np.array([0,0,120]), upper_white=np.array([180,50,255])) :
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # BGR -> HSV
    mask = cv2.inRange(hsv, lower_white, upper_white)   # 흰색범위 지정한 마스크 생성
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)   # 원본 이미지에 마스크 적용, 흰색 영역만 남김
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY) # 그레이스케일
    return gray

def gray_to_canny(gray, threshold=100):
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morphed, 40, 120, apertureSize=3)
    return edges

def edges_to_lines(edges, minLineLength=30, maxLineGap=30):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines

def draw_lines(edges, lines, color = (0,255,0), thickness=2):
    pre_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(pre_frame, (x1, y1), (x2, y2), (0,255,0), 2)

    return pre_frame

def get_center_from_lines(lines, y=380):
    if lines is None or len(lines) == 0:
        return None

    x_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # y를 선분이 통과하면
        if (y1 - y) * (y2 - y) <= 0:
            if y2 != y1:
                x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                x_candidates.append(int(x_at_y))
            elif y1 == y2 == y:  # 수평선
                x_candidates.extend([x1, x2])

    if x_candidates:
        center_x = int(np.mean(x_candidates))
        return center_x
    else:
        print(" y = 380에서 선분을 찾지 못했습니다.")
        return 320

def get_motor_angle(center_x, img_width=640): # 아직, PID로 변환하는 과정은 없음
    # x좌표를 0~180도로 매핑
    angle = int(center_x * 180 / img_width)
    angle = max(0, min(180, angle)) # 0~180도 범위 제한
    return angle