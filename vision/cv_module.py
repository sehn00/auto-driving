# 차선 인식을 위한 영상 전처리 및 중심좌표 계산

import cv2
from networkx import edges
import numpy as np

lower_white= np.array([0,0,120])
upper_white = np.array([180,50,255])
vertices = np.array([[ # 마스킹할 영역을 정의
    (100, 480),        # 좌하
    (280, 300),               # 좌상
    (360, 300),               # 우상
    (540, 480)]], dtype=np.int32) # 우하

def warp_image(image, srt_pts, dst_size=(640, 480)):    # 버드아이뷰
    width, height = dst_size
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(srt_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, dst_size)
    return warped

def origin_to_gray(frame) :
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # BGR -> HSV
    mask = cv2.inRange(hsv, lower_white, upper_white)   # 흰색범위 지정한 마스크 생성
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)   # 원본 이미지에 마스크 적용, 흰색 영역만 남김
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY) # 그레이스케일
    return gray

def gray_to_canny(gray):
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) # 이진화
    kernel = np.ones((5, 5), np.uint8)  # 모폴로지 필터용 5x5 커널 생성
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # 모폴로지 연산
    edges = cv2.Canny(morphed, 40, 120, apertureSize=3) # Canny Edge 검출, apertureSize가 크면 더 부드러운 엣지
    return edges 

def edges_to_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    return lines

def draw_lines(frame, lines, color = (0,255,0), thickness=2):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1,y1),(x2,y2), color, thickness)
    return frame

def get_center_from_lines(lines, y=380, img_width=640):
    """
    HoughLinesP의 lines 배열에서 y=380을 지나는 각 직선의 x좌표들을 구해서 중심 x좌표를 반환
    """
    if lines is None or len(lines) == 0:
        return None

    x_candidates = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # y=380을 선분이 통과하는지 확인
        if (y1 - y) * (y2 - y) > 0:
            # y=380보다 위쪽/아래쪽에 모두 있음(통과 안 함)
            continue

        # y=380일 때의 x좌표 구하기 (선분 공식 이용)
        if y2 != y1:
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            x_candidates.append(x_at_y)
        # 만약 수평선(y1==y2==y)이면, 양 끝점 x값 모두 후보에 추가
        elif y1 == y2 == y:
            x_candidates.extend([x1, x2])

    if len(x_candidates) == 0:
        return None

    center_x = int(np.mean(x_candidates))
    result_x = center_x - img_width // 2
    return result_x

def get_motor_angle(center_x, img_width=640):
    # x좌표를 0~180도로 매핑
    angle = int(center_x * 180 / img_width)
    angle = max(0, min(180, angle)) # 0~180도 범위 제한
    return angle
