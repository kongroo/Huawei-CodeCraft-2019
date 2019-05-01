# -*- coding: utf-8 -*-
import os
import cv2
import string
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from model_service.pytorch_model_service import PTServingBaseService
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)
logging.info('start')

num_classes = 43
num_labels = 9
index = {"深": 0, "秦": 1, "京": 2, "海": 3, "成": 4, "南": 5, "杭": 6, "苏": 7, "松": 8,
            "0": 9, "1": 10, "2": 11, "3": 12, "4": 13,"5": 14, "6": 15, "7": 16, "8": 17, "9": 18,
            "A": 19,"B": 20, "C": 21, "D": 22, "E": 23, "F": 24, "G": 25, "H": 26, "J": 27, "K": 28,
            "L": 29, "M": 30, "N": 31, "P": 32, "Q": 33, "R": 34, "S": 35, "T": 36, "U": 37, "V": 38,
            "W": 39, "X": 40, "Y": 41, "Z": 42};
chars = ["深", "秦", "京", "海", "成", "南", "杭", "苏", "松", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"];

class lpr_service(PTServingBaseService):
    def __init__(self, model_name, model_path):
        super(lpr_service, self).__init__(model_name, model_path)
        device = torch.device('cpu')
        self.model = models.resnet34(num_classes=num_classes*num_labels)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.load_state_dict(torch.load(os.path.join(self.model_path), map_location=device))
        self.model.eval()
        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #self.normalize = transforms.Normalize(mean=[0.315, 0.351, 0.474], std=[0.232, 0.228, 0.181])
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.Resize((70, 356)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _preprocess(self, data):
        logging.info("__preprocess__")
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                image = image.convert('RGB')
                image = self.transform(image)
                break
        return image[None]

    def _postprocess(self, probs):
        logging.info("_postprocess")
        probs = probs.view(num_classes, num_labels)
        labels = torch.argmax(probs, dim=0).numpy()
        pred = ''.join([chars[i] if i >= 9 else str(i) for i in labels])
        return pred

    def _inference(self, data):
        logging.info("_inference")
        return self.model(data)
