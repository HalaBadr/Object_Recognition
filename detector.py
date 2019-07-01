import cv2 as cv
cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'faster_rcnn_resnet50_coco_2018_01_28.pbtxt')
out=[]
class detector :

	def get_objects(self, img):

		boxes = self.__get_objects_positions(detector,img)
		res = []
		for box in boxes:
			object = self.__get_object(detector,img, box)
			res.append((object, box))
		return res

	def __get_objects_positions(self, img):

		cvNet.setInput(cv.dnn.blobFromImage(img, size=(224, 224), swapRB=True, crop=False))
		cvOut = cvNet.forward()
		boxes = []
		rows = img.shape[0]
		cols = img.shape[1]
		for detection in cvOut[0, 0, :, :]:
			score = float(detection[2])
			if score > 0.3:
				#print(detection[0], " ", detection[1])
				left = detection[3] * cols
				top = detection[4] * rows
				right = detection[5] * cols
				bottom = detection[6] * rows
				plt = (left, top)
				prb = (right, bottom)
				boxes.append((plt, prb))
		return boxes

	def __get_object(self, img, box):

		plt, prb = box
		object = img[int(plt[1]):int(prb[1]),int(plt[0]):int(prb[0])]
		return object
