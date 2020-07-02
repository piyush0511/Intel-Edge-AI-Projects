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

class Facedet:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.5, extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
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
            results = self.net.requests[0].outputs[self.output_name]
            self.image = self.preprocess_output(results, image)
        return self.image

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        return image

    def preprocess_output(self, results, image):
        for obj in results[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3] * self.initial_w)
                ymin = int(obj[4] * self.initial_h)
                xmax = int(obj[5] * self.initial_w)
                ymax = int(obj[6] * self.initial_h)
                self.image = image[ymin:ymax,xmin:xmax]
                #cv2.rectangle(frame, (10, 10), (50, 50), (0, 55, 255), 4)

        return self.image
