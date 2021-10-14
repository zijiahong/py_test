
from platform import machine
import cv2
import numpy as np


sift = cv2.xfeatures2d.SIFT_create()
img1  = cv2.imread("test1.png")
(kp1, des1) = sift.detectAndCompute(img1,None)   

img2  = cv2.imread("test2.jpg")
(kp2, des2) = sift.detectAndCompute(img2,None)   

# 获取flann匹配器
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
# 进行匹配
matches = flann.knnMatch(des1, des2, k=2)


good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(kp2[m.trainIdx])
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
cv2.rectangle(img2, (min[0], min[1]), (max[0], max[1]), (0, 255, 0), 2)
cv2.imshow("img2",img2)
cv2.waitKey(0)