import cv2
import numpy as np

lower_white= np.array([0,0,150])
upper_white = np.array([180,50,255])
vertices = np.array([[ # 마스킹할 영역을 정의
    (100, 480),        # 좌하
    (280, 300),               # 좌상
    (360, 300),               # 우상
    (540, 480)]], dtype=np.int32) # 우하

def ROI(frame, vertices): # ROI를 설정 
    mask = np.zeros_like(frame)
    
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]  # 3: RGB, 4: RGBA
        ignore_mask_color = (255,) * channel_count  # ex. (255, 255, 255)
    else:
        ignore_mask_color = 255  # 흑백일 경우

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_frame = cv2.bitwise_and(frame, mask)

    return masked_frame    

def warp_image(image, vertices, dst_size=(640, 480)):
    width, height = dst_size
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(vertices, dst_pts)
    warped = cv2.warpPerspective(image, M, dst_size)
    return warped

def pre_image(frame) :
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morphed, 40, 120)
    return edges


def get_center(edges, y=390):
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