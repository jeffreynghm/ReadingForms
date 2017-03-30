import cv2
import numpy as np
import pandas as pd

img = cv2.imread('2004FLG-6.png', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
#blurred = cv2.GaussianBlur(img, (5, 5), 0)
kernel = np.ones((5,5), np.uint8)
#blurred = cv2.dilate(img, kernel, iterations=5)
blurred = cv2.erode(img, kernel, iterations=10)
##datadict_pred_path= '/home/bnpp/ReadingForms/01_ObservatoryData/Blocks_Pred/2017-03-29_12:32:46/dataDict_pred.txt'
##dataframe = pd.read_csv(datadict_pred_path, delimiter='\t')
##for row in dataframe.iterrows():
##        #print(row)
##        xCor = int(row[1]['bbCoord_x0'])
##        yCor = int(row[1]['bbCoord_y0'])
##        xCor1 = int(row[1]['bbCoord_x1'])
##        yCor1 = int(row[1]['bbCoord_y1'])
##        cv2.rectangle(img,(xCor,yCor),(xCor1,yCor1),color=(255,255,255),thickness=-1)
        
sobel_horizontal = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)



cv2.imshow('Original', img)
cv2.imshow('blurred', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical)

cv2.waitKey(0)
