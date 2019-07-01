import glob
import cv2 as cv
import os
from recognizer import *
images=[]

#read images from folder

#path = "E:\Hala\MyComputer\BOOKS\programming\Python\Object_Recognition\Test_Images"
path = "E:\Hala\MyComputer\BOOKS\programming\Python\Object_Recognition\predict"
data_path = os.path.join(path,'*g')
files = glob.glob(data_path)
print(path)
for file in files:
    a = cv.imread(file)

    image = mark_objects(a,file)
    print("After mark_objects()")
    images.append(image)

for img in images :
    print("before imshow()")
    cv.imshow('img', img)
    cv.waitKey(10000)
    print("After imshow()")
    cv.destroyAllWindows()
