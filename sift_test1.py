from platform import machine
import cv2
import numpy as np


sift = cv2.xfeatures2d.SIFT_create()
img1  = cv2.imread("test1.png")
(kp1, des1) = sift.detectAndCompute(img1,None)   

img2  = cv2.imread("test2.jpg")
(kp2, des2) = sift.detectAndCompute(img2,None)   

# print(np.array(kp1[0]))
fsw = cv2.FileStorage('a.yml', cv2.FILE_STORAGE_WRITE)
fsw.write("des1",des1)
fsw.release()
fs = cv2.FileStorage('a.yml', cv2.FILE_STORAGE_READ)
des1 = fs.getNode("des").mat()
print(des1)


# bf = cv2.BFMatcher()
# matches  = bf.knnMatch(des1,des2,k=2)
# good = []
# for m,n in matches:
#     if m.distance < 0.75 * n.distance:
#         good.append([m])

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

# cv2.rectangle(img2,(xmin,ymin),(xmax,ymax),(0, 255, 0), 2)
# cv2.imshow("img",img2)
# cv2.waitKey(0)






