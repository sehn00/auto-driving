import cv2
import numpy as np

# 이미지 파일 경로
image_path = 'test_pic_2.jpg'    # 상대경로 (pwd: E:\dev_in_E\git-clones-E\auto_driving> )
# test_pic_1.jpg / test_pic_2.jpg / image_of_test.PNG

frame = cv2.imread(image_path)
frame = cv2.resize(frame, (640, 480))
if frame is None:
    print(f"이미지를 불러올 수 없습니다: {image_path}")
    exit()

# 트랙바/창 생성 (lower_white, upper_white, binary)
cv2.namedWindow('lower_white')
# cv2.moveWindow('lower_white', -1000, -1000)     # 창 안보이게 치우기

cv2.namedWindow('upper_white')
# cv2.moveWindow('upper_white', -1000, -1000)     # 창 안보이게 치우기

cv2.createTrackbar('lower_S', 'lower_white', 0, 255, lambda x: None)
cv2.createTrackbar('lower_V', 'lower_white', 180, 255, lambda x: None)
cv2.createTrackbar('upper_S', 'upper_white', 28, 255, lambda x: None)

cv2.namedWindow('binary')
cv2.createTrackbar('thresh', 'binary', 175, 255, lambda x: None)
# cv2.moveWindow('binary', -1000, -1000)          # 창 안보이게 치우기

# hough lines 창 생성
cv2.namedWindow('hough')
cv2.createTrackbar('minLineLength', 'hough', 30, 100, lambda x: None)
cv2.createTrackbar('maxLineGap', 'hough', 30, 100, lambda x: None)
# cv2.moveWindow('hough', -1000, -1000)           # 창 안보이게 치우기

# 표시용 창 생성 및 크기 조절 가능
for name in ['original_image', 'hough_lines_only', 'original_with_lines', 'processed_frame']:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 480)


while True:
    # 트랙바에서 값 읽기
    ls = cv2.getTrackbarPos('lower_S', 'lower_white')   # 인수 2개: '트랙바 이름', '트랙바 창 이름' -> 트랙바 값 반환
    # lv = cv2.getTrackbarPos('lower_V', 'lower_white')   # lower_V = 얼마나 밝아야 흰색으로 볼 것인가
    lv = 196                                            # 수정: 196 임의 설정
    us = cv2.getTrackbarPos('upper_S', 'upper_white')   # upper_S = 얼마나 채도가 낮아야 흰색으로 볼 것인가
    lower_white = np.array([0, ls, lv])
    upper_white = np.array([180, us, 255])

    # 추가 4. 흰색&검은색만 남기기 ---------------------------------------------------
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 흰색 범위 (채도 낮고, 밝기 높음)
    lower_white_full = np.array([0, 0, 180])
    upper_white_full = np.array([180, 40, 255])
    mask_white_only = cv2.inRange(hsv_full, lower_white_full, upper_white_full)

    # 검은색 범위 (밝기 낮음)
    lower_black_full = np.array([0, 0, 0])
    upper_black_full = np.array([180, 255, 50])
    mask_black_only = cv2.inRange(hsv_full, lower_black_full, upper_black_full)

    # 흰색 + 검은색 마스크 합치기
    mask_wb = cv2.bitwise_or(mask_white_only, mask_black_only)

    # 마스크 적용
    frame = cv2.bitwise_and(frame, frame, mask=mask_wb)
    # -----------------------------------------------------------------------------

    # 추가 4. 흰/검만 남기는 전처리 (Lab 기반, 극단 필터) -------------------------------
    # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    # L, A, B = cv2.split(lab)

    # # 무채색(그레이/흰/검): a,b가 128 근처
    # ta, tb = 10, 10            # a,b 허용 범위(작을수록 더 극단)
    # mask_a = cv2.inRange(A, 128 - ta, 128 + ta)
    # mask_b = cv2.inRange(B, 128 - tb, 128 + tb)
    # mask_neutral = cv2.bitwise_and(mask_a, mask_b)

    # # 밝기 기준으로 흰/검 분리
    # L_white_min = 200          # 높일수록 더 순백만 통과
    # L_black_max = 60           # 낮출수록 더 순흑만 통과
    # mask_white = cv2.inRange(L, L_white_min, 255)
    # mask_black = cv2.inRange(L, 0, L_black_max)

    # # 무채색 조건과 AND
    # mask_white = cv2.bitwise_and(mask_white, mask_neutral)
    # mask_black = cv2.bitwise_and(mask_black, mask_neutral)

    # # 최종: 흰 + 검만 남김
    # mask_wb = cv2.bitwise_or(mask_white, mask_black)

    # # (선택) 작은 점 제거
    # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # mask_wb = cv2.morphologyEx(mask_wb, cv2.MORPH_OPEN, kernel, iterations=1)

    # # 원본 프레임에 마스크 적용 → 나머지 색 전부 제거
    # frame = cv2.bitwise_and(frame, frame, mask=mask_wb)

    # # 두 톤으로 보고 싶으면(디버깅용):
    # # result = np.zeros_like(frame)
    # # result[mask_white > 0] = (255, 255, 255)
    # # result[mask_black > 0] = (0, 0, 0)
    # # cv2.imshow('wb_only', result)
    # ----------------------------------------------------------------------------



    # 1. HSV 마스킹 및 gray 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)

    # 추가 1. 미디언 블러 (반사광 억제)
    gray = cv2.medianBlur(gray, 3)

    # 추가 2-1. 화이트 탑햇 (작은 반사광 추출해서 제거)
    kernel_size = 7  # 차선 두께보다 약간 작은 크기로 조정
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # 추가 2-2. 반사광 억제: 원본 gray에서 tophat 결과 빼기
    gray = cv2.subtract(gray, tophat)

    # 2. binary threshold/canny
    threshold_val = cv2.getTrackbarPos('thresh', 'binary')
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morphed, 80, 200, apertureSize=5)         # 수정 3. Canny 강한 경계 검출 (기존: 40, 120, 3)

    # 3. 허프 라인 검출
    minLineLength = cv2.getTrackbarPos('minLineLength', 'hough')
    maxLineGap = cv2.getTrackbarPos('maxLineGap', 'hough')
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=minLineLength, maxLineGap=maxLineGap)

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

    # (4) 원본 이미지만 따로
    cv2.imshow('original_image', frame)
    cv2.imshow('hough_lines_only', hough_img)
    cv2.imshow('original_with_lines', original_with_lines)
    cv2.imshow('processed_frame', pre_frame)

    # y = 380 에서 차선 중심 계산 및 x_candidates 저장
    y_target = 380  # 원하는 y좌표

    x_candidates = []

    if lines is not None:
     for line in lines:
            x1, y1, x2, y2 = line[0]
         # y=380을 선분이 지나갈 경우만!
            if (y1 - y_target) * (y2 - y_target) <= 0:
             # 선분 공식으로 y=380에서 x 좌표 계산
                if y2 != y1:
                    x_at_y = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
                    x_candidates.append(int(x_at_y))
                elif y1 == y2 == y_target:  # 수평선(드물지만)
                   x_candidates.extend([x1, x2])

    if x_candidates:
        mean_x = int(np.mean(x_candidates))
        cv2.circle(hough_img, (mean_x, y_target), 8, (0, 0, 255), -1)  # 빨간 점 표시

    cv2.imshow('hough_img', hough_img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break


cv2.destroyAllWindows()



"""
사진 저장 코드
        frame = runtime.camera.get_image()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)        
        if frame is None:
            print("❌ 프레임 없음, 저장 안 함")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"frame_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            ok = cv2.imwrite(filepath, frame)
            if ok:
                print(f"✅ 이미지 저장 완료: {filename}")
            else:
                print(f"❌ 이미지 저장 실패: {filename}")

        time.sleep(0.3)       

"""