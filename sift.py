from platform import machine
import cv2
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()


def get_matchs(des, img):
    res = []
    (kp, img_des) = sift.detectAndCompute(img, None)
    for d in des:
        matches = bf.knnMatch(d, img_des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(kp[m.trainIdx])
        good = cv2.KeyPoint_convert(good)
        min = [1000000000, 1000000000]
        max = [0, 0]
        for i in good:
            if max[0] < i[0]:
                max[0] = i[0]
            if max[1] < i[1]:
                max[1] = i[1]
            if min[0] > i[0]:
                min[0] = i[0]
            if min[1] > i[1]:
                min[1] = i[1]
        cv2.rectangle(img, (min[0], min[1]), (max[0], max[1]), (0, 255, 0), 2)
        # minx | miny | maxx | maxy
        res.append((min[0], min[1], max[0], max[1]))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return res


def append_img_des(img, name, file_name):
    kp, des = sift.detectAndCompute(img, None)
    fsw = cv2.FileStorage(file_name, cv2.FILE_STORAGE_APPEND)
    fsw.write(name, des)
    fsw.release()
    return name

def get_des(names,file_name):
    res = []
    fs = cv2.FileStorage(file_name,cv2.FileStorage_READ)
    for name in names:
        des =  fs.getNode(name).mat()
        if des != None:
            res.append(des)
            continue
        print("node not found, name is ",name)
    return res

def center_points(list):
    res = []
    for l in list:
        x = (l[2] + l[0])/2
        y = (l[3] + l[1])/2
        res.append((x, y))
    return res

img1 = cv2.imread("test1.png")
img2 = cv2.imread("test2.jpg")

kp1, des1 = sift.detectAndCompute(img1, None)
print(get_matchs([des1], img2))
print(center_points(get_matchs([des1], img2)))
