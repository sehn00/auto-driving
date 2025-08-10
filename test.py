import cv2
import numpy as np

# 이미지 파일 경로
image_path = 'C:\\Users\\HeoYun\\Desktop\\auto_driving\\image_of_test.PNG'
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (640, 480))
if frame is None:
    print(f"이미지를 불러올 수 없습니다: {image_path}")
    exit()

# 모든 트랙바를 한 창에 만들기
cv2.namedWindow('all')
cv2.createTrackbar('lower_S', 'all', 0, 255, lambda x: None)
cv2.createTrackbar('lower_V', 'all', 180, 255, lambda x: None)
cv2.createTrackbar('upper_S', 'all', 28, 255, lambda x: None)
cv2.createTrackbar('thresh', 'all', 175, 255, lambda x: None)
cv2.createTrackbar('minLineLength', 'all', 30, 100, lambda x: None)
cv2.createTrackbar('maxLineGap', 'all', 30, 100, lambda x: None)

# 표시용 창 생성 및 크기 조절 가능
for name in ['original_image', 'hough_lines_only', 'original_with_lines', 'processed_frame']:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 480)

while True:
    # 트랙바 값 읽기
    ls = cv2.getTrackbarPos('lower_S', 'all')
    lv = cv2.getTrackbarPos('lower_V', 'all')
    us = cv2.getTrackbarPos('upper_S', 'all')
    threshold_val = cv2.getTrackbarPos('thresh', 'all')
    minLineLength = cv2.getTrackbarPos('minLineLength', 'all')
    maxLineGap = cv2.getTrackbarPos('maxLineGap', 'all')

    lower_white = np.array([0, ls, lv])
    upper_white = np.array([180, us, 255])

    # 1. HSV 마스킹 및 gray 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)

    # 2. binary threshold/canny
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morphed, 40, 120, apertureSize=3)

    # 3. 허프 라인 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

    # (1) 허프 라인만 그린 이미지 (검정 바탕)
    hough_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_img, (x1, y1), (x2, y2), (0,255,0), 2)

    # (2) 원본 이미지 위에 허프 라인 그린 이미지
    original_with_lines = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(original_with_lines, (x1, y1), (x2, y2), (0,255,0), 2)

    # (3) processed_frame(엣지에 허프 라인 그리기)
    pre_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(pre_frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # (4) 창에 표시
    cv2.imshow('original_image', frame)
    cv2.imshow('hough_lines_only', hough_img)
    cv2.imshow('original_with_lines', original_with_lines)
    cv2.imshow('processed_frame', pre_frame)

    # y=380에서 중심 계산
    y_target = 380
    x_candidates = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (y1 - y_target) * (y2 - y_target) <= 0:
                if y2 != y1:
                    x_at_y = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
                    x_candidates.append(int(x_at_y))
                elif y1 == y2 == y_target:
                    x_candidates.extend([x1, x2])
    if x_candidates:
        mean_x = int(np.mean(x_candidates))
        cv2.circle(hough_img, (mean_x, y_target), 8, (0, 0, 255), -1)

    cv2.imshow('hough_img', hough_img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
