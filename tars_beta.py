from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  scipy.ndimage import morphology
import cv2
import whale_preprocess as whale
import copy
import covariance_descriptor as cov_mat
import time
from tqdm import tqdm
import os

path = "/media/sai/New Volume1/Practice/whale/face"
face = [];
non_face = [];
count = 0;
for dirName, subdirList, fileList in os.walk(path):
	for fname in fileList:
			count  = count + 1;
			print count;
                        a = path + '/' + fname;
			object_image = cv2.imread(a,1);
			x,y = object_image[:,:,0].shape;
			feat_matrix = np.zeros((x,y,7),dtype="float64")
			feat_matrix[:,:,0:3] = object_image;
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,1,0)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,1,0)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,1,0)
			M_x = np.square(R) + np.square(G) + np.square(B);
			M_x = np.sqrt(M_x);
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,0,1)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,0,1)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,0,1)
			M_y = np.square(R) + np.square(G) + np.square(B);
			M_y = np.sqrt(M_y);
  			feat_matrix[:,:,3] = M_x;
  			feat_matrix[:,:,4] = M_y;
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,2,0)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,2,0)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,2,0)
			M_x = np.square(R) + np.square(G) + np.square(B);
			M_x = np.sqrt(M_x);
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,0,2)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,0,2)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,0,2)
			M_y = np.square(R) + np.square(G) + np.square(B);
			M_y = np.sqrt(M_y);
  			feat_matrix[:,:,5] = M_x;
  			feat_matrix[:,:,6] = M_y;		
			object_integral = cov_mat.get_integral_mat(feat_matrix);
			object_corr = cov_mat.get_corr_mat(feat_matrix);
			object_cov = cov_mat.get_cov_mat(object_corr,object_integral,(0,0),(x-1,y-1));
			face.append(object_cov);
path = "/media/sai/New Volume1/Practice/whale/non_face"
for dirName, subdirList, fileList in os.walk(path):
	for fname in fileList:
			count  = count + 1;
			print count;
                        a = path + '/' + fname
			object_image = cv2.imread(a,1);	
			x,y = object_image[:,:,0].shape;
			feat_matrix = np.zeros((x,y,7),dtype="float64")
			feat_matrix[:,:,0:3] = object_image;
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,1,0)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,1,0)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,1,0)
			M_x = np.square(R) + np.square(G) + np.square(B);
			M_x = np.sqrt(M_x);
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,0,1)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,0,1)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,0,1)
			M_y = np.square(R) + np.square(G) + np.square(B);
			M_y = np.sqrt(M_y);
  			feat_matrix[:,:,3] = M_x;
  			feat_matrix[:,:,4] = M_y;
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,2,0)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,2,0)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,2,0)
			M_x = np.square(R) + np.square(G) + np.square(B);
			M_x = np.sqrt(M_x);
			B = cv2.Sobel(object_image[:,:,0],cv2.CV_64F,0,2)
			G = cv2.Sobel(object_image[:,:,1],cv2.CV_64F,0,2)
			R = cv2.Sobel(object_image[:,:,2],cv2.CV_64F,0,2)
			M_y = np.square(R) + np.square(G) + np.square(B);
			M_y = np.sqrt(M_y);
  			feat_matrix[:,:,5] = M_x;
  			feat_matrix[:,:,6] = M_y;		
			object_integral = cov_mat.get_integral_mat(feat_matrix);
			object_corr = cov_mat.get_corr_mat(feat_matrix);
			object_cov = cov_mat.get_cov_mat(object_corr,object_integral,(0,0),(x-1,y-1));
			non_face.append(object_cov);
face = np.asarray(face);
non_face = np.asarray(non_face);
np.save('/media/sai/New Volume1/Practice/whale/files/face.npy', face);
np.save('/media/sai/New Volume1/Practice/whale/files/non_face.npy', non_face);
