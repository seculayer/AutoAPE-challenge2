import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


# 이미지 리사이징
def resize_with_padding(img, img_size, img_id, dataset):

    #  가로가 세로보다 길 경우
    if (img.shape[1] > img.shape[0]):
        percent = img_size/img.shape[1]
    else:
        percent = img_size/img.shape[0]


    try :
        img = cv2.resize(img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)

    except cv2.error as e :
        reload_img = cv2.imread(f'./dataset/{dataset}/{img_id}.jpg')
        img_trim = reload_img[12:-40, :, :]
        img = cv2.resize(img_trim, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)


    y,x,h,w = (0, 0, img.shape[0], img.shape[1])

    w_x = (img_size-(w-x))/2
    h_y = (img_size-(h-y))/2

    if(w_x < 0):
        w_x = 0
    elif(h_y < 0):
        h_y = 0

    #  x축으로 w_x, y축으로 h_y 이동
    M = np.float32([[1,0,w_x], [0,1,h_y]])
    img_re = cv2.warpAffine(img, M, (img_size, img_size))

    return img_re


#  엣지 부분만 크롭
def auto_cropping (img_id, dataset):

    if dataset == 'train_images' :
        img = cv2.imread(f'./dataset/{dataset}/{img_id}.jpg')
        img_trim = img[12:-40, :, :]
        return img_trim


    with open('./dataset/iWildCam_2019_Annotations/iWildCam_2019_CCT_Bboxes.json', 'r') as f:
        json_data = json.load(f)

    json_anno = json_data['annotations']
    img_info = list(filter(lambda x:x["image_id"]==img_id,json_anno))
    try :
        category_info = img_info[-1].get("category_id")

    except IndexError as e:
        print("No such id's information")
        category_info = 0

    img = cv2.imread(f'./dataset/{dataset}/{img_id}.jpg')
    img_trim = img[12:-40, :, :]

    # plt.imshow(img)
    # plt.title("original image")
    # plt.show()
    #
    # plt.imshow(img_trim)
    # plt.title("up down bars trimmed image")
    # plt.show()

    if category_info != 0 :
        img_gray = cv2.cvtColor(img_trim, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0)

        ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        thresh_edged = cv2.Canny(thresh, 10, 250)
        # edged = cv2.Canny(blur, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        closed = cv2.morphologyEx(thresh_edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        mg_trim = img_trim[y:y+h, x:x+w]

        # plt.imshow(blur)
        # plt.title("blurred image")
        # plt.show()

        # plt.imshow(thresh)
        # plt.title("thresh binary")
        # plt.show()

        # plt.imshow(thresh_edged)
        # plt.title("thresh_edged")
        # plt.show()

        # plt.imshow(edged)
        # plt.title("canny - edge")
        # plt.show()

        # plt.imshow(closed)
        # plt.title("closed image")
        # plt.show()

        # plt.imshow(mg_trim)
        # plt.title("trim completed")
        # plt.show()

        return mg_trim

    else :
        return img_trim


#  이미지 밝기 수정 및 엣지 추출
def img_pre_processing (img_id, dataset, img_size=224):
    wb = cv2.xphoto.createSimpleWB()
    wb.setP(0.4)

    trimmed_img = auto_cropping(img_id, dataset)

    img_wb = wb.balanceWhite(trimmed_img)
    img_lab = cv2.cvtColor(img_wb, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    lab_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    resized = resize_with_padding(lab_img, img_size, img_id, dataset)

    # plt.imshow(resized)
    # plt.title("CLAHE & WB & Resized")
    # plt.show()

    return resized


#  테스트 이미지 전처리
def test_img_pre_processing (img_id, dataset, img_size=224):
    wb = cv2.xphoto.createSimpleWB()
    wb.setP(0.4)

    img = cv2.imread(f'./dataset/{dataset}/{img_id}.jpg')
    trimmed_img = img[12:-40, :, :]

    img_wb = wb.balanceWhite(trimmed_img)
    img_lab = cv2.cvtColor(img_wb, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    lab_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    resized = resize_with_padding(lab_img, img_size, img_id, dataset)

    return resized