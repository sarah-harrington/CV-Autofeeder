import cv2
import time
from ultralytics import YOLO

# set our time between checks
delay = 7

# cat detection
catLabel = 15

# confidence standard
goalConf = 0.35

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# open our video feed
cam = None
for i in range(10):
	cam = cv2.VideoCapture(i)
	
	# check to see if it worked
	suc, frame = cam.read()
	if suc:
		break

if cam == None:
	print("Unable to open cam")
	exit()
	
last_run_time = time.time()
while True:
	
	# Read in image from cam
	suc, image = cam.read()
	
	# only progress if the read was successful
	if suc:
	
		# if our delay is up
		if time.time() - last_run_time > delay:
			# Perform object detection on an image using the model
			results = model([image])
			
			# get confidence and labels arrays
			confs = results[0].boxes.conf.numpy()
			labels = results[0].boxes.cls.numpy()
			
			print(f"labels: {labels}")
			
			if catLabel in labels:
				print("cat")
			
			# check for confidence it's actually a cat	
			for i, label in enumerate(labels):
				if label == catLabel:
					if confs[i] > goalConf:
						print(f"{confs[i]}% sure it's a cat")
					else:
						print(f"probably not a cat: {confs[i]}")

			
			# print(results.classes)
			last_run_time = time.time()
		
		cv2.imshow("test", image)
		#print(image.shape)
		cv2.waitKey(1)
		
