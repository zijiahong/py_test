from platform import machine
import cv2
import numpy as np


img1  = cv2.imread("test1.png")
img2  = cv2.imread("test2.jpg")

sift = cv2.xfeatures2d.SIFT_create()
(kp1, des1) = sift.detectAndCompute(img1,None)  
(kp2, des2) = sift.detectAndCompute(img2,None)   

# cv2.drawKeypoints(img1, kp1, img1)
# cv2.imshow("img1",img1)
# # cv2.waitKey(0)



# cv2.drawKeypoints(img2, kp2, img2)
# cv2.imshow("img2",img2)
# cv2.waitKey(0)

bf = cv2.BFMatcher()
matches  = bf.knnMatch(des1,des2,k=2)


# trainIdx  在 kp list 中的数组坐标
good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

print(len(good))
if len(good) < 15:
    print("关联性不强")
else:
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    cv2.imshow("img3",img3)
    cv2.waitKey(0)



# kp3 = []
# for i in good:
#     kp3.append(kp2[i[0].trainIdx])
# points2f = cv2.KeyPoint_convert(kp3) 
# xmin = 1000000000
# ymin = 1000000000 
# xmax = 0
# ymax = 0

# for i in points2f:
#     if xmin <i[0]:
#         xmax = i[0]
#     if ymax < i[1]:
#         ymax = i[1]
#     if xmin > i[0]:
#         xmin = i[0]
#     if ymin > i[1]:
#         ymin = i[1]

# print(xmin,ymin)
# print(xmax,ymax)

# cv2.rectangle(img2,(xmin,ymin),(xmax,ymax),(0, 255, 0), 2)
# cv2.imshow("img",img2)
# cv2.waitKey(0)


# print(good)






# fs = cv2.FileStorage('abc.yml', cv2.FileStorage_WRITE)
# fs.write("kp1",kp1)
# fs.write("des1",des1)


# cv2.FileStorage_APPEND()