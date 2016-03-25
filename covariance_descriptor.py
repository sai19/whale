
'''
test code
'''

from __future__ import division
from scipy.linalg import eigvalsh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  scipy.ndimage import morphology
import cv2
import copy

def get_integral_mat(feature_mat):
	x,y,z = feature_mat.shape;
	feature_mat = feature_mat.astype('float64');
	IntTensor = np.zeros(feature_mat.shape,dtype='float64');
	for feat in range(z):
		feature_mat[:,:,feat] = feature_mat[:,:,feat]/np.max(feature_mat[:,:,feat]); 
	IntTensor[0,0] = feature_mat[0,0];
	for i in range(x):
		for j in range(y):
			if (i>0) or (j>0):
				if(i==0):
					IntTensor[i,j] = IntTensor[i,j-1] + feature_mat[i,j];
				elif (j==0):  
					IntTensor[i,j] = IntTensor[i-1,j] + feature_mat[i,j];
				elif (i>0) and (j>0):
					IntTensor[i,j] = IntTensor[i-1,j] + IntTensor[i,j-1] - IntTensor[i-1,j-1]+feature_mat[i,j];

	return (IntTensor);	


def get_corr_mat(feature_mat):
	x,y,z = feature_mat.shape;
	corr_mat = np.zeros((x,y,z,z),dtype='float64');
	feature_mat = feature_mat.astype('float64');
	for feat in range(z):
		feature_mat[:,:,feat] = feature_mat[:,:,feat]/np.max(feature_mat[:,:,feat]); 
	for a in range(z):
		for b in range(z-a):
			mat = feature_mat[:,:,a]*feature_mat[:,:,b+a];
			corr_mat[0,0,a,b+a] = mat[0,0];
			for i in range(x):
				for j in range(y):
					if (i>0) or (j>0):
						if(i==0):
							corr_mat[i,j,a,b+a] = corr_mat[i,j-1,a,b+a] + mat[i,j];
						elif (j==0):  
							corr_mat[i,j,a,b+a] = corr_mat[i-1,j,a,b+a] + mat[i,j];
						elif (i>0) and (j>0):
							corr_mat[i,j,a,b+a] = corr_mat[i-1,j,a,b+a] + corr_mat[i,j-1,a,b+a] - corr_mat[i-1,j-1,a,b+a]+mat[i,j];
						if (b!=0):
							corr_mat[i,j,b+a,a] = corr_mat[i,j,a,b+a];
	return (corr_mat);

def get_cov_mat(corr_mat,IntTensor,upper_point,lower_point):
	x2,y2 = lower_point;
	x1,y1 = upper_point;
	n = (x2-x1)*(y2-y1);	
	CorrMat = corr_mat[x2,y2] + corr_mat[x1,y1] - corr_mat[x1,y2] - corr_mat[x2,y1];
	M1 =  IntTensor[x2,y2] + IntTensor[x1,y1] - IntTensor[x1,y2] - IntTensor[x2,y1];
	M2 = np.transpose(M1);
	IntMat = np.outer(M1,M2);
	CovMat = (CorrMat-(IntMat/n))/(n-1);
	return(CovMat);	

def distance(M1,M2):
	if(M1.shape!=M2.shape):
		print "Error : M1,M2 are not of the same size"
	else:	
		x,y = M1.shape
		if(x!=y):
			print "A square matrix is expected"
		else:
			eigen = eigvalsh(M1,M2);
			log = np.log(eigen);
			log = np.square(log);
			distance = np.sum(log);
			distance = np.sqrt(distance); 	
	return (distance);

def get_feat_mat(object_image):
	if(len(object_image.shape)>2):	
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
	else : 
		print "image was expected"
		feat_matrix = None;
	return(object_image); 
