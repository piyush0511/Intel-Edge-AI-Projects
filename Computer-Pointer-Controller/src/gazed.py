'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

class Gazedet:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, val, imagel,imager):
        input_imgl , input_imgr = self.preprocess_input(imagel,imager)
        input_dict = {"left_eye_image":input_imgl , "right_eye_image":input_imgr , "head_pose_angles":val}
        self.net.start_async(request_id=0,inputs=input_dict)
        status = self.net.requests[0].wait(-1)
        if status == 0:
            results = self.net.requests[0].outputs[self.output_name]
        return np.asarray(np.squeeze(results))

    def preprocess_input(self, imagel, imager):
        n, c, h, w = 1,3,60,60
        imagel = cv2.resize(imagel, (w, h))
        imagel = imagel.transpose((2, 0, 1))
        imagel = imagel.reshape((n, c, h, w))
        imager = cv2.resize(imager, (w, h))
        imager = imager.transpose((2, 0, 1))
        imager = imager.reshape((n, c, h, w))        
        return imagel, imager

