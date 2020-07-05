
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import numpy as np
from faced import Facedet
from input_feeder import InputFeeder
from posed import Posedet
from landed import Landet
from gazed import Gazedet
from mouse_controller import MouseController
import math

import logging


def draw_arrow(frame, cord, l, r, results):
	lf = (cord[0] + l[0], cord[1] + l[1])
	rf = (cord[0] + r[0], cord[1] + r[1])
	cv2.arrowedLine(frame, lf, (cord[0] + l[0] + int(results[0]*300),cord[1] + l[1] + int(-results[1]*300)), (155, 0, 0), 3)
	cv2.arrowedLine(frame, rf, (cord[0] + r[0] + int(results[0]*300),cord[1] + r[1] + int(-results[1]*300)), (255,0, 0), 3)
	return frame



def build_camera_matrix(center_of_face, focal_length):
	cx = int(center_of_face[0])
	cy = int(center_of_face[1])
	camera_matrix = np.zeros((3, 3), dtype='float32')
	camera_matrix[0][0] = focal_length
	camera_matrix[0][2] = cx
	camera_matrix[1][1] = focal_length
	camera_matrix[1][2] = cy
	camera_matrix[2][2] = 1
	return camera_matrix




def draw_axes(frame, center_of_face, yaw, pitch, roll):
	focal_length = 950.0
	scale = 50
	yaw *= np.pi / 180.0
	pitch *= np.pi / 180.0
	roll *= np.pi / 180.0
	cx = int(center_of_face[0])
	cy = int(center_of_face[1])
	Rx = np.array([[1, 0, 0],
			[0, math.cos(pitch), -math.sin(pitch)],
			[0, math.sin(pitch), math.cos(pitch)]])
	Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
			[0, 1, 0],
			[math.sin(yaw), 0, math.cos(yaw)]])
	Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
		[math.sin(roll), math.cos(roll), 0],
		[0, 0, 1]])
	# ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
	R = Rz @ Ry @ Rx
	camera_matrix = build_camera_matrix(center_of_face, focal_length)
	xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
	yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
	zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
	zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
	o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
	o[2] = camera_matrix[0][0]
	xaxis = np.dot(R, xaxis) + o
	yaxis = np.dot(R, yaxis) + o
	zaxis = np.dot(R, zaxis) + o
	zaxis1 = np.dot(R, zaxis1) + o
	xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
	yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
	p2 = (int(xp2), int(yp2))
	cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
	xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
	yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
	p2 = (int(xp2), int(yp2))
	cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
	xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
	yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
	p1 = (int(xp1), int(yp1))
	xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
	yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
	p2 = (int(xp2), int(yp2))
	cv2.line(frame, p1, p2, (255, 0, 0), 2)
	cv2.circle(frame, p2, 3, (255, 0, 0), 2)
	cv2.imshow("frame",frame)
	return frame




def main(args):

	logging.basicConfig(level=logging.INFO, 
	format="%(asctime)s [%(levelname)s] %(message)s", 
	handlers=[logging.FileHandler("gaze.log"), logging.StreamHandler()])

	modelf=args.modelf
	modelp=args.modelp
	modell=args.modell
	modelg=args.modelg
	device=args.device
	threshold=args.threshold


	### Load the model through ###
	logging.info("============== Models Load time ===============") 
	start_time = time.time()
	fa= Facedet(modelf, device, threshold)
	fa.load_model()
	pa= Posedet(modelp, device)
	pa.load_model()
	la= Landet(modell, device)
	la.load_model()
	ga= Gazedet(modelg, device)
	ga.load_model()
	logging.info("Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )
	logging.info("==============  End =====================") 
	m= MouseController("low", "fast")


	if args.video is not None:
		cap=cv2.VideoCapture(args.video)
	else:
		cap=cv2.VideoCapture(0)


	initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	time.sleep(3)

	try:
		while True:
			
			ret,frame = cap.read()

			if not ret:
				continue	
			frame = cv2.flip(frame,1)
			t=time.time()

			# get face 
			image , c, cord = fa.predict(frame)

			# get head pose estimation
			val = pa.predict(image).reshape((1,3))

			# get landmarks
			li , ri, l, r = la.predict(image)

			# get gaze vector
			results = ga.predict(val, li, ri)
			logging.info("Total Inference Time:{:.1f}ms".format(1000*(time.time()-t)))

			# move mouse
			m.move(results[0], results[1])
			yaw = val[0][0]
			pitch = val[0][1]
			roll = val[0][2]
			

			# display results and frame
			if args.flag == "yes":
				#ref: https://knowledge.udacity.com/questions/171017
				frame = draw_axes(frame, c, yaw, pitch, roll)
				frame = draw_arrow(frame, cord, l, r,results)
				cv2.imshow("frame",frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		cv2.destroyAllWindows()
		cap.release()

	except Exception as ex:
		logging.exception("Error in inference:" + str(ex))




if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('--modelf', required=True, help="Face Model" )
	parser.add_argument('--modelp', required=True, help="Head Pose Model")
	parser.add_argument('--modell', required=True, help="Landmarks Model")
	parser.add_argument('--modelg', required=True, help="Gaze Model")
	parser.add_argument('--flag', default="None", help=" Display Face (if yes type 'yes')")
	parser.add_argument('--device', default='CPU', help="Device")
	parser.add_argument('--video', default=None, help="Path to the video file")
	parser.add_argument('--threshold', default=0.60, help="Threshold")
    
	args=parser.parse_args()

	main(args)
