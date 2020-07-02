
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import numpy as np
from faced import Facedet
from input_feeder import InputFeeder

def main(args):
	model=args.model
	device=args.device
	threshold=args.threshold

	fa= Facedet(model, device, threshold)
	fa.load_model()

	if args.video is not None:
		cap=cv2.VideoCapture(args.video)
	else:
		cap=cv2.VideoCapture(0)
	
	while True:

		ret,frame = cap.read()

		if not ret:
			continue	

		frame = fa.predict(frame)
		#cv2.imshow("frame",frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	cv2.destroyAllWindows()
	cap.release()

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('--model', required=True)
	parser.add_argument('--device', default='CPU')
	parser.add_argument('--video', default=None)
	parser.add_argument('--threshold', default=0.60)
    
	args=parser.parse_args()

	main(args)
