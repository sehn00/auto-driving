# 차선 인식을 위한 영상 전처리 및 중심좌표 계산

import cv2
import numpy as np

lower_white= np.array([0,0,200])
upper_white = np.array([180,50,255])
src_pts = np.array([[170, 290], [440, 290], [564, 390], [80, 390]], dtype=np.float32)

def warp_image(image, src_pts, dst_size=(640, 480)):    # 버드아이뷰
    width, height = dst_size
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, dst_size)
    return warped

def pre_image(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # BGR -> HSV
    mask = cv2.inRange(hsv, lower_white, upper_white)   # 흰색범위 지정한 마스크 생성
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)   # 원본 이미지에 마스크 적용, 흰색 영역만 남김
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY) # 그레이스케일
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) # 이진화
    kernel = np.ones((5, 5), np.uint8)  # 모폴로지 필터용 5x5 커널 생성
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # 모폴로지 연산
    edges = cv2.Canny(morphed, 40, 120) # Canny Edge 검출
    return edges 

def get_center(edges, y=390):   # 차선 중심 위치 반환
    height, width = edges.shape

    if y >= height:
        raise ValueError(f"y={y}는 이미지 높이({height})보다 클 수 없습니다.")

    # y=390 줄에서 흰색(255) 픽셀 x 위치 찾기
    white_x_indices = np.where(edges[y] == 255)[0]  # 1D array of x positions

    if len(white_x_indices) == 0:
        return None  # 흰 선이 없음

    # 중심 x 좌표 계산
    center_x = int(np.mean(white_x_indices))
    result_x = center_x - 320
    return center_x, result_x