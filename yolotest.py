from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import numpy as np 
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1723728383<0 else 0
import torch
import pyttsx3
from threading import Thread

probability_threshold=0.7

def SpeakText(command):
	
	# Initialize the engine
	engine = pyttsx3.init()
	engine.say(command)
	engine.runAndWait()


model = YOLO('best2.pt')
'''
img = cv2.imread('dataset/test/currency_test.jpg')
results = model.predict(img)
print(results)
objects = []
for r in results:
	annotator = Annotator(img)
	boxes = r.boxes
	for box in boxes:
		b = box.xyxy[0].to(dtype=torch.float)  # get box coordinates in (left, top, right, bottom) format
		c = box.cls
		#print(model.names[int(c)])
		annotator.box_label(b, model.names[int(c)])
		objects.append(model.names[int(c)])
		
img = annotator.result()
print(objects)  
#SpeakText(objects)
cv2.imshow('YOLO V8 Detection', img)     
cv2.waitKey(0)
#cv2.destroyAllWindows()


'''
cap = cv2.VideoCapture(0)
#cap.set(3, 640)
#cap.set(4, 480)

def detectCurrency():
	while True:
		objects = []
		_, img = cap.read()
		
		# BGR to RGB conversion is performed under the hood
		# see: https://github.com/ultralytics/ultralytics/issues/2575
		results = model.predict(img)

		for r in results:
			annotator = Annotator(img)
			boxes = r.boxes
			for box in boxes:
				b = box.xyxy[0].to(dtype=torch.float)  # get box coordinates in (left, top, right, bottom) format
				c = box.cls
				confidence = box.conf  # Get confidence score

				# Check if confidence is above the threshold
				if confidence > probability_threshold:
					annotator.box_label(b, model.names[int(c)])
					objects.append(model.names[int(c)])
					
				#print(model.names[int(c)])
				#annotator.box_label(b, model.names[int(c)])
				#objects.append(model.names[int(c)])
			
		img = annotator.result()
		print(objects)  
		SpeakText(objects)
		#t = Thread(target=SpeakText,args=(objects,))
		#t.deamon = True
		#t.start()

		#cv2.imshow('YOLO V8 Detection', img)
		imgencode=cv2.imencode('.jpg',img)[1]
		stringData=imgencode.tostring()
		yield (b'--frame\r\n'
			b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
		'''		     
		if cv2.waitKey(1) & 0xFF == ord(' '):
			break
		if cv2.waitKey(1) & 0xFF == ord('s'):
			SpeakText(objects)
		'''

	cap.release()
	cv2.destroyAllWindows()
