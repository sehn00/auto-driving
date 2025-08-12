import cv2
import numpy as np

# PID 제어 변수 (전역)
pid_last_error = 0
pid_integral = 0

def origin_to_gray(frame, lower_white=np.array([0,0,196]), upper_white=np.array([180,28,255])) :
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)    # BGRA -> RGB
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)   # 흰색범위 지정한 마스크 생성
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)   # 원본 이미지에 마스크 적용, 흰색 영역만 남김
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY) # 그레이스케일
    gray = cv2.medianBlur(gray, 3) # 미디언 블러 (반사광 억제)
    kernel_size = 7  # 차선 두께보다 약간 작은 크기로 조정
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    gray = cv2.subtract(gray, tophat) # 반사광 억제: 원본 gray에서 tophat 결과 빼기

    return gray

def gray_to_canny(gray, threshold=175):
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morphed, 40, 120, apertureSize=3) # sobel = 3x3
    return edges

import numpy as np

def get_center_from_canny(edges, min_cluster_gap=15):
    """
    Canny 엣지 이미지의 여러 가로줄을 스캔하여 안정적인 차선 중심을 찾습니다.
    
    Args:
        edges (numpy.ndarray): gray_to_canny()의 결과 이미지 (0 또는 255 값)
        min_cluster_gap (int): 같은 차선으로 판단할 픽셀 간의 최대 거리

    Returns:
        int: 감지된 여러 차선 중심점의 평균 x좌표.
        None: 차선을 감지하지 못한 경우.
    """
    h, w = edges.shape[:2]

    # 스캔할 y좌표들을 이미지 높이의 65% ~ 85% 사이에서 5개 선택합니다.
    # 이 범위는 주행 환경에 따라 조절할 수 있습니다.
    scan_ys = np.linspace(int(h * 0.7), int(h * 0.9), num=5, dtype=int)
    
    valid_centers = []
    
    # 선택된 각 y좌표에 대해 차선 중심을 찾습니다.
    for y in scan_ys:
        # 1. 해당 라인의 모든 엣지(흰색 픽셀) x좌표를 찾습니다.
        xs = np.where(edges[y] == 255)[0]
        if xs.size < 2:
            continue  # 엣지가 2개 미만이면 다음 라인으로 넘어갑니다.

        # 2. 연속된 픽셀들을 하나의 클러스터(차선 덩어리)로 묶습니다.
        clusters = []
        start_x = xs[0]
        for i in range(1, len(xs)):
            if xs[i] - xs[i-1] > min_cluster_gap:
                clusters.append((start_x, xs[i-1]))
                start_x = xs[i]
        clusters.append((start_x, xs[-1]))

        # 3. 각 클러스터의 중심 x좌표를 계산합니다.
        cluster_centers = [(s + e) // 2 for s, e in clusters]

        # 4. 클러스터(차선)가 2개 이상 감지되면, 가장 바깥쪽 두 개의 중심으로 차선 중심을 계산합니다.
        if len(cluster_centers) >= 2:
            left_lane_x = cluster_centers[0]
            right_lane_x = cluster_centers[-1]
            center_x = (left_lane_x + right_lane_x) // 2
            valid_centers.append(center_x)

    # 5. 유효하게 찾은 중심점들의 평균을 최종 center_x로 반환합니다.
    if not valid_centers:
        return None  # 유효한 중심점을 하나도 찾지 못한 경우

    return int(np.mean(valid_centers))



def get_motor_angle(center_x, last_angle, max_step=10, img_width=640):
    if center_x is None:
        return last_angle  # 검출 안 되면 그대로 유지

    # center_x → 0~180도로 매핑
    target_angle = int(center_x * 180 / img_width)
    target_angle = max(0, min(180, target_angle))

    # 변화량 제한
    if target_angle > last_angle + max_step:
        new_angle = last_angle + max_step
    elif target_angle < last_angle - max_step:
        new_angle = last_angle - max_step
    else:
        new_angle = target_angle

    return new_angle

def detect_black_yellow(frame, x1, y1, x2, y2):
    direction = None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    H, W = frame.shape[:2]
    # 1) ROI 클리핑(경계 밖 방지)
    x1 = max(0, min(W-1, int(x1)))
    x2 = max(0, min(W-1, int(x2)))
    y1 = max(0, min(H-1, int(y1)))
    y2 = max(0, min(H-1, int(y2)))
    roi = frame[y1:y2, x1:x2].copy()

    # 2) 전처리 + HSV 변환
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    # 3) 색 범위(환경에 따라 조정 가능)
    # 검정: 밝기 V가 매우 낮은 영역
    lower_black = np.array([0,   0,   0],   dtype=np.uint8)
    upper_black = np.array([180, 255, 60],  dtype=np.uint8)   # V<=60 정도

    # 노랑: H≈20~35, 채도/밝기 충분히 높은 영역
    lower_yellow = np.array([20,  80,  80], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

    # 4) 마스크 생성
    mask_black  = cv2.inRange(hsv, lower_black,  upper_black)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5) 노이즈 정리(모폴로지)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_black  = cv2.morphologyEx(mask_black,  cv2.MORPH_OPEN, k, iterations=1)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, k, iterations=1)

    # 6) 비율 계산
    area = (y2 - y1) * (x2 - x1)
    black_ratio  = float(mask_black.sum()  / 255) / max(1, area)
    yellow_ratio = float(mask_yellow.sum() / 255) / max(1, area)

    if yellow_ratio > black_ratio:
        print('left')
        return 30
    else : # black_ratio >= yellow_ratio
        print('right')  
        return 150
        