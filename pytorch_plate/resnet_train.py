# -*- coding: utf-8 -*-
import os
import os
import cv2
import string
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import argparse
import moxing as mox
import logging
import sys

mox.file.shift('os', 'mox')
sys.path.insert(0, "../../python")
sys.setrecursionlimit(1000000)
random.seed(2019)
torch.manual_seed(2019)

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

class GaussBulr(object):
    def __init__(self, radius=2):
        self.radius = radius
    def __call__(self,img):
        return img.filter(ImageFilter.GaussianBlur(random.randint(1, self.radius)))

class AddGaussNoise(object):
    def __init__(self, mean=0, std=2):
        self.mean = mean
        self.std = std
    def __call__(self,img):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        noise = np.zeros(img.shape, dtype=np.uint8)
        mean = random.random() * self.mean
        std =  random.random() * self.std
        cv2.randn(noise, (mean, mean, mean), (std, std, std))
        img = img + noise
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, test=False):
        self.root_dir = root_dir
        self.data = open(os.path.join(root_dir, 'train-data-label.txt'), encoding='utf-8').readlines()
        #normalize = transforms.Normalize(mean=[0.315, 0.351, 0.474], std=[0.232, 0.228, 0.181])
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    #GaussBulr(2),
                    #AddGaussNoise(),
                    #transforms.RandomAffine(degrees=2,translate=(0.01,0.01),shear=2),
                    transforms.Resize((70, 356)),
                    #transforms.RandomAffine(degrees=10,translate=(0.01,0.01),scale=(0.8,1.0),shear=10),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((70, 356)),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transform
        if not test:
            random.seed(2019)
            random.shuffle(self.data)
            if train:
                self.data = self.data[:int(0.9*len(self.data))]
            else:
                self.data = self.data[int(0.9*len(self.data)):]
        self.data = [line.split(',') for line in self.data]
        self.data = [[image.strip(), label.strip()] for image, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'train-data', self.data[idx][1])
        with Image.open(img_name) as img:
            image = img.convert('RGB')
        image = self.transform(image)
        label = np.array([index[i] for i in self.data[idx][0]])
        label = torch.from_numpy(label)
        return image, label

def accuracy(output, target):
    #batch_size * num_class * num_label
    #bath_size * num_label
    hit = 0
    count = 0
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        hit += torch.sum(torch.eq(pred, target)).item()
        count += batch_size * num_labels
    return hit, count

def validate(val_loader, model, criterion):
    model.eval()
    losses = 0.0
    total_hit = 0
    total_cnt = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            output = output.view(-1, num_classes, num_labels)
            loss = criterion(output, target)
            losses += loss.item()
            hit, count = accuracy(output, target)
            total_hit += hit
            total_cnt += count
    logging.info("validate loss : %.6f acc : %.6f" % (losses, 1.0 * total_hit / total_cnt))
    return 1.0 * total_hit / total_cnt

def train(args, my_data):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    logging.info('start with arguments %s', args)
    lr=0.001
    batch_size = 16
    epochs = 600
    model = models.resnet34(num_classes=num_classes*num_labels)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
    #scheduler_update = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10.0)
    model.cuda()
    criterion.cuda()
    train_dataset = LicensePlateDataset(my_data)
    val_dataset = LicensePlateDataset(my_data, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=8)
    best_acc = 0.0
    stop_counter = 0
    for epoch in range(epochs):
        model.train()
        #scheduler.step()
        #if epoch == 10:
        #    scheduler_update.step()
        losses = 0.0
        total_hit = 0
        total_cnt = 0
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            output = output.view(-1, num_classes, num_labels)
            loss = criterion(output, target)
            #print(target[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            losses += loss
            hit, count = accuracy(output, target)
            total_hit += hit
            total_cnt += count
            if i % 50 == 0:
                logging.info('[Epoch %d, Batch %5d] loss: %.6f acc: %.6f' % (epoch + 1, i + 1, loss, 1.0 * hit / count))
        logging.info("train loss : %.6f acc : %.6f" % (losses, 1.0 * total_hit / total_cnt))
        acc = validate(val_loader, model, criterion)
        if acc >= best_acc:
            if acc >= 0.97:
                torch.save(model.state_dict(), os.path.join(args.train_url, 'model-resnet34-epoch%d.pth' % epoch))
            best_acc = acc
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter == 10 and best_acc >= 0.98667:
                break
    logging.info("best acc : %.6f" % best_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="license plate recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 运行时参数，在线训练默认会传入data_url参数，指示训练的数据集路径，也就是在训练作业中选定的数据所在文件夹
    parser.add_argument('--data_url', type=str, default='s3://obs-car-reg', help='the training data')

    # 运行时参数，在线训练默认会传入train_url参数，指示训练的输出的模型所在路径，也就是在训练作业中选定的模型输出文件夹
    parser.add_argument('--train_url', type=str, default='s3://obs-car-reg//model', help='the path model saved')
    args, unkown = parser.parse_known_args()

    # 复制OSB路径文件至当前容器/cache/my_data目录下，后续操作直接使用/cache/my_data进行数据引用
    mox.file.copy_parallel(src_url= args.data_url, dst_url='/cache/my_data')
    my_data = '/cache/my_data'
    train(args=args, my_data=my_data)
