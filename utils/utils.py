import cv2
import numpy as np
from PIL import Image

def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    last_point = []
    point_coord_sets = []
    point_labels_sets = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img
        if event == cv2.EVENT_LBUTTONDOWN:
            last_point = [x, y]
            point_coord_sets.append(last_point)
            point_labels_sets.append(1)
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            last_point = [x, y]
            point_coord_sets.append(last_point)
            point_labels_sets.append(0)
            cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
        elif event == cv2.EVENT_MBUTTONDOWN:
            keep_looping = False
        cv2.imshow("image", img)

    cv2.setMouseCallback("image", mouse_callback)
    while keep_looping:
        cv2.waitKey(1) 

    cv2.destroyAllWindows()

    return point_coord_sets, point_labels_sets

def get_box_point(img_path):
    # TODO 获取左上角和右下角的坐标位置，确定选定区域
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    
    last_point = []
    keep_looping = True
    
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point


def get_brush_point(img_path):
    # TODO 获取刷子的坐标位置，及时处理
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    
    last_point = []
    keep_looping = True
    
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return last_point
    