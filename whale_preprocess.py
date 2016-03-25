from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  scipy.ndimage import morphology
import cv2
import copy


def subimage(image, centre, theta, width, height):
	mapping = cv2.getRotationMatrix2D(centre,theta,1.0)
	image = cv2.warpAffine(image,mapping,image[:,:,0].shape,flags=cv2.INTER_LINEAR)
	output_image = cv2.getRectSubPix(image,(height,width),centre)   
	return output_image

def rotateImage(image, angle):
  image_center = tuple(np.array(image[:,:,0].shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image[:,:,0].shape,flags=cv2.INTER_LINEAR)
  return result

def extract_whale(im1):
	image = cv2.medianBlur(im1,151);
	image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);
	kernel = np.ones((50,50),dtype='float32')/2500;
	mean = cv2.filter2D(image,-1,kernel);
	mean = mean.astype('float32');
	im = image.astype('float32');
	mean = im - mean;
	mean = abs(mean);
	mean = mean.astype('uint8');
	ret, mean = cv2.threshold (mean,np.median(mean), 255, cv2.THRESH_BINARY);
	#mean =  cv2.medianBlur(mean,51);
	#ret, mean = cv2.threshold (mean,np.median(mean), 255, cv2.THRESH_BINARY);
	mean = morphology.binary_fill_holes(mean);
	mean = morphology.binary_closing(mean,iterations=10);
	#mean = morphology.binary_opening(mean,iterations=10);
	mean = mean.astype('uint8');
	m = copy.copy(mean)
	_,contours, hierarchy = cv2.findContours(mean,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
	area = 0;
	mask = np.zeros(mean.shape,np.uint8)		
	for cnt in contours:
		if(cv2.contourArea(cnt)>=area):
			area = cv2.contourArea(cnt);
			large = cnt;
	cv2.drawContours(mask,[large],0,255,-1);
	rect = cv2.minAreaRect(large);
	center,(w,h),t = rect;	
	if(t<-45):
			t = t + 90;		
	mask = np.zeros(mean.shape,dtype='uint8');					
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(mask,[box],0,255,-1);
	image = cv2.bitwise_and(im1,im1,mask=mask);
	image = rotateImage(image,t+90);
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);
	ret,gray = cv2.threshold (gray,0, 255, cv2.THRESH_BINARY);
	_,contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
	area = 0;
	mask = np.zeros(gray.shape,np.uint8)		
	for cnt in contours:
		if(cv2.contourArea(cnt)>=area):
			area = cv2.contourArea(cnt);
			large = cnt;
	x,y,w,h = cv2.boundingRect(large);
	image = image[y:y+h,x:x+w];
	return(m,image,t);
