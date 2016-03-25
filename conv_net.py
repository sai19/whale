from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from keras.datasets import mnist
import cv2
import os
from pandas import DataFrame
from collections import OrderedDict


path = "/media/sai/New Volume1/Practice/whale/face_detector/face/"
p1 = "/media/sai/New Volume1/Practice/whale/face_detector/face1/"
for dirName, subdirList, fileList in tqdm(os.walk(path)):
        for fname in fileList:
			#print fname;
                        a = path + '/' + fname
                        image = cv2.imread(a,1);
			name = fname.split(".")[0]
			output_path = p1 + name + ".png"
			cv2.imwrite(output_path,image);
path = "/media/sai/New Volume1/Practice/whale/face_detector/non_face/"
p1 = "/media/sai/New Volume1/Practice/whale/face_detector/non_face1/"
for dirName, subdirList, fileList in tqdm(os.walk(path)):
        for fname in fileList:
                        a = path + '/' + fname
                        image = cv2.imread(a,1)
			name = fname.split(".")[0]
			output_path = p1 + name + ".png"
			cv2.imwrite(output_path,image);
