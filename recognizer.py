
import Model
import cv2
from detector import *

def extract_objects(image,file):

    items= []
    objects = detector.get_objects(detector, image)
    m = Model.ObjectsModel(100)
    if len(objects)==0 :
        objectName = m.predict(image)
    else :
     objectName = m.predict(objects)
    for i in range (len(objects)) :
        item = []

        item.append(objects[i][0]) #object
        item.append(objects[i][1]) #positions of object

       # item.append(objectName)
        items.append(item)
    return objectName ,items

def mark_objects(image,file):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 51, 51)
    color = (66,206,244)
    offest_x = 2
    offest_y = 5

    objectName ,extracted_objects = extract_objects(image,file)
    print("After extract_objects()")

    for i in range(len(extracted_objects)):
        tmp = (int(extracted_objects[i][1][0][0]-offest_x),
               int(extracted_objects[i][1][0][1]-offest_y))

        image = cv2.rectangle(image,(int(extracted_objects[i][1][0][0]),int(extracted_objects[i][1][0][1]))
        ,(int(extracted_objects[i][1][1][0]) ,int(extracted_objects[i][1][1][1] ))
        ,(66,206,244),2)

        image = cv2.putText(image,
                            objectName[i],
                            tmp,
                            font,
                            font_scale,
                            font_color,
                            2)

    return image