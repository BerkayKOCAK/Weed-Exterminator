# USAGE
# python real_time_object_detection.py --config yolov3.cfg --weights yolov3.weights --classes coco.names

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2




# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 288      #Width of network's input image
inpHeight = 288      #Height of network's input image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument('-c', '--config', required=True,help = 'path to yolo config file');
ap.add_argument('-w', '--weights', required=True, help = 'path to yolo pre-trained weights');
ap.add_argument('-cl', '--classes', required=True,help = 'path to text file containing class names');
ap.add_argument("-con", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.4,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]

	# Scan through all the bounding boxes output from the network and keep only the
	# ones with high confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])
				
 
	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

	for i in indices:
		
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
	# Draw a bounding box.
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))
	
	label = '%.2f' % conf
		 
	# Get the label for the class name and its confidence
	if classes:
		assert(classId < len(classes))
		label = '%s:%s' % (classes[classId], label)
		#print(classId)
	#Display the label at the top of the bounding box
	labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])
	cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

classes = None
with open(args.get("classes"), 'r') as f:
	classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# load our serialized model(weights) from disk
print("[INFO] loading model...");
net = cv2.dnn.readNetFromDarknet( args.get("config"),args.get("weights"));


# initialize the video stream, allow the cammera sensor to warmup,

print("[INFO] starting video stream...")
cap = VideoStream(src=0).start()
#cap = cv2.VideoCapture(0)
# Get the video writer initialized to save the output video

while cv2.waitKey(1) < 0:
	 
	# get frame from the video
	frame = cap.read()
	frame = imutils.resize(frame, width=400)
   
	
 
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
 
	# Sets the input to the network
	net.setInput(blob)
 
	# Runs the forward pass to get output of the output layers
	
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	outs = net.forward(ln)
	# Remove the bounding boxes with low confidence
	postprocess(frame, outs)
	
	# Put efficiency information. The function getPerfProfile returns the 
	# overall time for inference(t) and the timings for each of the layers(in layersTimes)
	t, _ = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
   
	
	cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
   
	cv2.imshow("Frame", frame)
 
   

# do a bit of cleanup
