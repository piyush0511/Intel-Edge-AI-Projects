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

class Posedet:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.initial_w = ''
        self.initial_h = ''
        self.image=[]
        
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

    def predict(self, image):
        self.initial_w = image.shape[1]
        self.initial_h = image.shape[0]
         
        input_img = self.preprocess_input(image)
        input_dict = {self.input_name:input_img}
        self.net.start_async(request_id=0,inputs=input_dict)
        status = self.net.requests[0].wait(-1)
        if status == 0:
            result1 = self.net.requests[0].outputs["angle_y_fc"]
            result2 = self.net.requests[0].outputs["angle_p_fc"]
            result3 = self.net.requests[0].outputs["angle_r_fc"]
        return np.asarray([np.squeeze(result1),np.squeeze(result2),np.squeeze(result3)])

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        return image


