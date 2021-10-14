import cv2
from a_start import A_Search,point

im = cv2.imread('image2.png')

# print("test",type(im))
# cv2.imshow("im",im)
# cv2.waitKey(0)
# cv2.sestroyAllWindows()



gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
# gray = cv2.resize(gray,(30,30))
gray = cv2.resize(gray,(int(im.shape[1]/30),int(im.shape[0]/30)))

# cv2.imshow("im",im)
# cv2.waitKey(0)
# cv2.sestroyAllWindows()

# cv2.imshow('gray',gray)
# cv2.waitKey(0)
# cv2.sestroyAllWindows()

str = '1123455780'

# print(gray)
map = []
person = []
target = []
for  i_index,i in gray:
    list = []
    for j_index,j in i:
        index = int((j/256)*len(str))
        if str[index] =='1':
            list.append(1)
        elif str[index] =='0':
            list.append(0)
        elif str[index] =='5':
            print(i_index)
            print(j_index)
            list.append(0)
            person.append(point(i_index,j_index))
        elif str[index] =='7':
            print(i_index)
            print(j_index)
            target.append(point(i_index,j_index))
    map.append(list)
print(len(map))

search = A_Search(person[0],target[0],map)

for i in search.process():
    pass

# print(search.close[0])
print(search.DisplayPath())