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
object_path = "/media/sai/New Volume1/Practice/whale/face/w_6665.jpg"
target_path = "/media/sai/New Volume1/Practice/whale/train/whale_24815/w_6665.jpg"
#data of the object image
object_image = cv2.imread(object_path,1);
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
#data of the target image
print time.clock();
print "object done"
target_image = cv2.imread(target_path,1);
#target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB);
x,y = target_image[:,:,0].shape;
feat_matrix = np.zeros((x,y,7),dtype="float64")
feat_matrix[:,:,0:3] = target_image;
B = cv2.Sobel(target_image[:,:,0],cv2.CV_64F,1,0)
G = cv2.Sobel(target_image[:,:,1],cv2.CV_64F,1,0)
R = cv2.Sobel(target_image[:,:,2],cv2.CV_64F,1,0)
M_x = np.square(R) + np.square(G) + np.square(B);
M_x = np.sqrt(M_x);
B = cv2.Sobel(target_image[:,:,0],cv2.CV_64F,0,1)
G = cv2.Sobel(target_image[:,:,1],cv2.CV_64F,0,1)
R = cv2.Sobel(target_image[:,:,2],cv2.CV_64F,0,1)
M_y = np.square(R) + np.square(G) + np.square(B);
M_y = np.sqrt(M_y);
feat_matrix[:,:,3] = M_x;
feat_matrix[:,:,4] = M_y;
B = cv2.Sobel(target_image[:,:,0],cv2.CV_64F,2,0)
G = cv2.Sobel(target_image[:,:,1],cv2.CV_64F,2,0)
R = cv2.Sobel(target_image[:,:,2],cv2.CV_64F,2,0)
M_x = np.square(R) + np.square(G) + np.square(B);
M_x = np.sqrt(M_x);
B = cv2.Sobel(target_image[:,:,0],cv2.CV_64F,0,2)
G = cv2.Sobel(target_image[:,:,1],cv2.CV_64F,0,2)
R = cv2.Sobel(target_image[:,:,2],cv2.CV_64F,0,2)
M_y = np.square(R) + np.square(G) + np.square(B);
M_y = np.sqrt(M_y);
feat_matrix[:,:,5] = M_x;
feat_matrix[:,:,6] = M_y;		
target_integral = cov_mat.get_integral_mat(feat_matrix);
target_corr = cov_mat.get_corr_mat(feat_matrix);
#searching for object
print time.clock();
print "object done"
print "done computing covariance matrices of object and target image"
x = 100;
y = 100;
cnt = 0;
x_t,y_t = target_image[:,:,0].shape
for size in range(17):
	x = int(x*(1.15));
        y = int(y*(1.15));
	X = int(x_t/50);
	Y = int(y_t/50);
	for i in range(X):
		for j in range(Y):
			x_u = 50*i;
			y_u = 50*j;
			x_l = min(50*i+x,x_t-1);
			y_l = min(50*j+y,y_t-1);
			cov = cov_mat.get_cov_mat(target_corr,target_integral,(x_u,y_u),(x_l,y_l));
			distance = cov_mat.distance(cov,object_cov);
			if(i+j==0):
				minimum = distance;
				min_image = target_image[x_u:x_l,y_u:y_l];
			elif(distance<=minimum):
				minimum = distance;
				min_image = target_image[x_u:x_l,y_u:y_l];	
min_image = cv2.cvtColor(min_image, cv2.COLOR_BGR2RGB);
print time.clock();
plt.figure();
plt.imshow(min_image);
plt.show()

