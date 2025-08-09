import cv2
import numpy as np

#lower_white= np.array([0,0,120])
#upper_white = np.array([180,50,255])


# PID 제어 변수 (전역)
pid_last_error = 0
pid_integral = 0

def origin_to_bev(frame): 
    width, height = 640, 480  # 출력 이미지 크기 고정
    dst_points = np.array([ # dst_point = 내가 펼친 사각형의 좌표
        [0, 0], # TL
        [width-1, 0], # TR
        [width-1, height-1], # BR
        [0, height-1] #
    ], dtype=np.float32)

    src_points = np.array([ # dst_point = 내가 펼친 사각형의 좌표
        [50, 300], # TL
        [590, 300], # TR
        [620, 380], # BR
        [20, 380] #
    ], dtype=np.float32)

    # 원근 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(frame, matrix, (width, height))
    
    return warped_image

def origin_to_gray(frame, lower_white=np.array([0,0,180]), upper_white=np.array([180,28,255])) :
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)    # BGRA -> RGB
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)   # 흰색범위 지정한 마스크 생성
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)   # 원본 이미지에 마스크 적용, 흰색 영역만 남김
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY) # 그레이스케일
    return gray

def gray_to_canny(gray, threshold=175):
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morphed, 40, 120, apertureSize=3)
    return edges

def edges_to_lines(edges, minLineLength=30, maxLineGap=30):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines

def draw_lines(frame, lines, color = (0,255,0), thickness=2):
    hough_img = np.zeros_like(frame)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_img, (x1,y1),(x2,y2), color, thickness)
    return hough_img

def get_center_from_lines(lines, y=190):
    if lines is None or len(lines) == 0:
        return None

    x_candidates = []
    # ✅ N×1×4 또는 N×4 형태 모두 안전하게 처리
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        # y를 선분이 통과하면
        if (y1 - y) * (y2 - y) <= 0:
            if y2 != y1:  # 기울어진 선
                x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                x_candidates.append(int(x_at_y))
            elif y1 == y2 == y:  # 수평선
                x_candidates.extend([x1, x2])

    if x_candidates:
        center_x = int(np.mean(x_candidates))
        return center_x
    else:
        return 320  # 후보가 없을 경우 중앙값 기본 반환

def get_motor_angle(center_x, img_width=640): # 아직, PID로 변환하는 과정은 없음
    # x좌표를 0~180도로 매핑
    if center_x is None:
        return 90
    angle = int(center_x * 180 / img_width)
    angle = int(max(0, min(180, angle))) # 0~180도 범위 제한
    return angle


def get_center_from_canny(canny_img, y=380):
    """
    Canny 이미지에서 특정 y좌표의 흰 픽셀의 x좌표 중심을 계산
    :param canny_img: Canny edge 이미지 (2D numpy array)
    :param y: 중심을 계산할 y 좌표 (default: 380)
    :return: 중심 x 좌표 (없으면 None)
    """
    if y >= canny_img.shape[0]:
        print(f"❌ y={y}는 이미지 높이({canny_img.shape[0]})를 벗어났습니다.")
        return None

    # 해당 y 좌표 라인의 모든 x 좌표에서 픽셀 값이 255인 (즉 흰색인) 위치 탐색
    white_x_positions = np.where(canny_img[y] == 255)[0]  # [0] 붙이면 x 좌표 배열만 반환
    white_x_list = white_x_positions.tolist()
    if len(white_x_positions) == 0:
        return None  # 중심 못 찾음

    center_x = int(np.mean(white_x_positions))
    return center_x


