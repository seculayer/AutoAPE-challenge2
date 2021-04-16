import numpy as np
import cv2


# 꽃 색깔 부분만 masking
def masking(file_name):

    img_color = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    lower = (80, 0, 0)
    upper = (120+10, 255, 255)

    img_mask = cv2.inRange(img_hsv, lower, upper)
    img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)

    img_file = cv2.resize(img_result, dsize=(225, 225))

    return img_file


# 꽃이 안핀 식물 - 엣지 뽑아내고 엣지 부분만 crop.
def tree_edge(file_name):

    img_color = cv2.imread(file_name, cv2.IMREAD_COLOR)

    blur = cv2.GaussianBlur(img_color, ksize=(3, 3), sigmaX=0)

    ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    thresh_edged = cv2.Canny(thresh, 50, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(thresh_edged, cv2.MORPH_CLOSE, kernel)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    erosion = cv2.erode(closed, k)

    img = img_color
    contours, _ = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_xy = np.array(contours)

    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)

    # y의 min과 max 찾기
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            #네번째 괄호가 0일때 x의 값
            value.append(contours_xy[i][j][0][1])
            y_min = min(value)
            y_max = max(value)

    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min

    img_trim = img[y:y+h, x:x+w]

    img_file = cv2.resize(img_trim, dsize=(225, 225))

    return img_file



