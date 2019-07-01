import glob
import cv2 as cv
import os
from detector import *
images=[]
out=[]
#read images from folder
class detectimage :

 path = "E:\Hala\MyComputer\BOOKS\programming\Python\Object_Recognition\predict"
 data_path = os.path.join(path,'*g')
 files = glob.glob(data_path)
 print(path)
 for file in files:
    a= cv.imread(file)
    images.append(a)

 for img in images :
    object = detector.get_objects(detector,img)
        #cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    i=0
    for obj in object:
        cv.imshow('img', obj[0])
        cv.waitKey(10000)
        cv.destroyAllWindows()
        i=i+1