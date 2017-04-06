#include <iostream>
#include <opencv2/opencv.hpp>

'''
v2This approach tries to identify the lines of the table, but the result is not very promising
v3: tries to find out the blocks of text
'''

#using namespace std;
import cv2
import numpy as np
import datetime
import os
from random import randint

timestampstr = "{:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
timestampstr=timestampstr+'/'
global stepSet
stepSet = []

def printPic(arr,stepSet,varname):
    
    cv2.imwrite(timestampstr+varname+'.jpg',arr)
    cv2.imshow(varname, arr)
    stepSet.append(varname)
    cv2.waitKey(0)
    return stepSet

##attempt to link all the boxes as a grid...but failed
def boxifyImg(img):
    shp = img.shape
    for h in range(0,shp[0]-1):
        startW = 0
        endW = 0
        for w in range(0,shp[1]-1):
            if h <= 1 or w <=1 :
                img[h][w] = 255
            else:
                if img[h][w] == 0:
                    if img[h][w-1] == 255:
                        startW = img[h][w]
                    if img[h][w+1] == 255:
                        endW = img[h][w]
                        img[h][startW:endW] = np.full((1,endW-startW+1),255)
                        startW = 0
                        endW = 0
    return img                
                
    

def downscale_image(im, H,W):
    """Shrink im until its longest dimension is <= max_dim.

    Returns new_image, scale (where scale <= 1).
    """
    shp = im.shape
    a =shp[0] #Height
    b =shp[1] #Width


    scaleW = 1.0 * W / b
    scaleH = 1.0 * H / a
    scale = min(scaleH,scaleW)
    

    new_im = cv2.resize(im,(W,H),interpolation = cv2.INTER_AREA)

    return new_im
def getLineDataset(lines):
    lineDB = np.array(shape=(len(lines),4),dtype = 'int32')
    i = 0
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(bw_Hough,(x1,y1),(x2,y2),150,2)
        lineDB[i][0] = x1
        lineDB[i][1] = y1
        lineDB[i][2] = x2
        lineDB[i][3] = y2
        i+=1
    return lineDB
        

def cannyHoughOp(bw_blurred):
    #remove the lines to black color
    #canny + hough
    tmpname = 'tmp'+str(randint(0,100))+'.jpg'

    cv2.imwrite(timestampstr+tmpname,bw_blurred)
    bw_blurred_r= cv2.imread(timestampstr+tmpname,0)
    bw_blurred_r = np.uint8(bw_blurred_r)
    
    bw_Canny = cv2.Canny(bw_blurred_r,50,150,apertureSize = 3)
    printPic(bw_Canny,stepSet,'bw_Canny')
    
    #no need o do hough for line
    Hough = True
    lines = []
    if Hough == True:
        bw_Hough=bw_Canny #keep a copy to avoid Canny being updated...after hough
        minLineLength = 10
        maxLineGap = 10
        
        lines = cv2.HoughLinesP(bw_Hough,1,np.pi/180,100,minLineLength,maxLineGap)
        print(str(len(lines))+' lines identfied')
        
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(bw_Hough,(x1,y1),(x2,y2),255,2)
        bw_out = bw_Hough
        printPic(bw_Canny,stepSet,'bw_Hough_Canny')
    else:
        bw_out = bw_Canny
    return bw_out, lines

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False
def connectnearContours(img):
    ret,thresh = cv2.threshold(img,127,255,0)
    img_con, contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i    
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    cv2.drawContours(img,unified,-1,(150),2)
    #cv2.drawContours(thresh,unified,-1,255,-1)
    cv2.imwrite(timestampstr+'group contours.jpg',img)
    cv2.imshow("group contours", img)
    stepSet.append("Contours")
    cv2.waitKey(0)
    return img

##def blurOp(bw,blurIter = 1):
##    ##remove noises
##    ##Apply morphology operations
##    kernel = np.zeros((2,2), np.uint8)
##    
##    #bw_blurred = cv2.dilate(bw, kernel = kernel, iterations=blurIter)
##
##    bw_blurred = cv2.GaussianBlur(bw, (5, 5), 0)
##    bw_blurred = cv2.dilate(bw, kernel = kernel, iterations=blurIter)
##    #bw_blurred = cv2.GaussianBlur(bw_blurred, (5, 5), 0)
##    #bw_blurred = cv2.erode(bw_blurred, kernel = kernel, iterations=blurIter)
##
##    printPic(bw_blurred,stepSet,'dilate')
##    return bw_blurred

def blurOp(bw,blurIter = 1,fontColor = "W",struct=False):
    ##remove noises
    ##Apply morphology operations
    if struct == True:
       kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,3)) 
    elif fontColor == "W":
        kernel = np.ones((2,2), np.uint8)
    else:
        kernel = np.zeros((2,2), np.uint8)    
   
    bw_blurred = cv2.dilate(bw, kernel = kernel, iterations=blurIter)

    printPic(bw_blurred,stepSet,'blur')
    
    return bw_blurred

def sharpenOp(bw,blurIter = 1,fontColor = "W",struct=False):
    ##remove noises
    ##Apply morphology operations
    if struct == True:
       k = cv2.getStructuringElement(cv2.MORPH_RECT,(17,3)) 
       #if fontColor == 'W':
           
    elif fontColor == "W":
         kernel = np.zeros((2,2), np.uint8)
    else:
        kernel = np.ones((2,2), np.uint8) 

    bw_blurred = cv2.erode(bw, kernel = kernel, iterations=blurIter)

    printPic(bw_blurred,stepSet,'sharpen')
    
    return bw_blurred
def morphOp(bw,fontColor = "W"):
    ##remove noises
    ##Apply morphology operations
    if fontColor == "W":
        kernel = np.ones((2,2), np.uint8)
    else:
        kernel = np.zeros((2,2), np.uint8) 

    closing = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    printPic(closing,stepSet,'MorpGrad')
    
    return closing

def morphOp2(bw_blurred,blurIter= 5):
    ##identify lines
    vertical = bw_blurred
    horizontal = bw_blurred
    scale = 100
    
    kernel = np.zeros((2,2), np.uint8)

#    for scale in range(1,501,10):
    horizontalsize = int(round(horizontal.shape[1]/scale))
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations=blurIter*2)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=blurIter*2)
    
    ##Show extracted horizontal lines
    printPic(horizontal,stepSet,"horizontal")
    
    useVertical = True
    if useVertical == True:
        scale = 50

        #scale +=5
        verticalsize = int(round(vertical.shape[1]/scale))
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize))
        vertical = cv2.erode(vertical, verticalStructure, iterations=blurIter*2)
        vertical = cv2.dilate(vertical, verticalStructure, iterations=blurIter*2)
        
        ##Show extracted vertical lines
        printPic(vertical,stepSet,"vertical")

        #mask
        mask = vertical + horizontal
        ##Show extracted vertical lines


        mask_erode = cv2.erode(mask, kernel = kernel, iterations=blurIter)
    else:
        mask_erode = horizontal
    printPic(mask_erode,stepSet,"dilate_erode_mask2")
      
       
    return mask_erode
def sobelOp(arr):
    #arr_out = cv2.Sobel(src=arr, ddepth=cv2.CV_64F, dx=1, dy=0)
    arr = cv2.Sobel(src=arr, ddepth=cv2.CV_64F, dx=1, dy=0)
    arr_out = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    printPic(arr_out,stepSet,'sobel')
    return arr_out

def contoursOpRect(bw_blurred):
    tmpname = 'tmp'+str(randint(0,100))+'.jpg'
    cv2.imwrite(timestampstr+tmpname,bw_blurred)
    bw_blurred_r= cv2.imread(timestampstr+tmpname,0)
    
    bw_blurred_r1, contours, hierarchy = cv2.findContours(bw_blurred_r
                                                          , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(bw_blurred_r,[box], -1, 150, 3)
    
def contoursOp(bw_blurred):

    bw_blurred_r2, contoursbb, hierarchy = cv2.findContours(bw_blurred
                                                            , cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(str(len(contoursbb)) +' bb contours found!')
    outbw = cv2.drawContours(bw_blurred_r2,contoursbb, -1, 255, 3)
    printPic(bw_blurred_r2,stepSet,'contours_bbox')
    
    return bw_blurred_r2,contoursbb

#put in a 2xn array who depict the x and y coordinates of the non-zero points
#return list of 1x2 array depicting the points
def nonZeroOp(arr):
    cnt = np.nonzero(arr)
    row = len(cnt[0])
    #print(cnt)
    i = 0
    pts = np.empty(shape=(row,2),dtype = 'int32')
    
    for i in range(row):
        pts[i][0] = cnt[0][i]
        pts[i][1] = cnt[1][i]

    return pts

def rotateOp(arr):
    
    points = nonZeroOp(arr)
    rect = cv2.minAreaRect(points)
    print(rect)
    box = cv2.boxPoints(rect)
    print(box)
    box = np.int0(box)

    #ret = drawn contours
    ret = cv2.drawContours(arr,[box],0,150,2)
    rectC =rect[0]
    rectSize =rect[1]
    recAng = rect[2]

    #
    Rotate = False
    if Rotate == True:
        rot_mat = cv2.getRotationMatrix2D(rectC,recAng,1.0)
        ret = cv2.warpAffine(ret, rot_mat, arr.shape,flags=cv2.INTER_LINEAR)
    #rotate the points?
    
    cv2.imwrite(timestampstr+'bwRotate.jpg',ret)
    cv2.imshow("bwRotate", ret)
    stepSet.append("bwRotate")
    cv2.waitKey(0)

    markBox = False #don't send the marked rectange 
    if markBox == True:
        arr = ret
        
    return arr 
def thresholdOp(arr,option=1):

    if option == 1:
        ret,arr_out = cv2.threshold(arr,127,255,cv2.THRESH_BINARY)
    elif option == 2:
        arr_out = cv2.adaptiveThreshold(arr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    printPic(arr_out,stepSet,'threshold')

    return arr_out

def main1():
    #path = '2004FLG-1.png'
    #path = './BL/Bill of Lading2Bill-of-Lading-Origin.jpg'
    path = './BL/Bill of Lading3Bill-of-lading-full.jpg'
    kernel = np.zeros((2,2), np.uint8)
    blurIter = 5
    #stepSet=[]
    os.makedirs(timestampstr)

    arr = cv2.imread(path)
    orgShape = arr.shape
    print(orgShape)
    
    printPic(arr,stepSet,'org')
    
    #resize image
    arr_resiz = downscale_image(arr, H=660,W=510)
    printPic(arr_resiz,stepSet,'resiz')

    #change to mono color
    if len(arr_resiz.shape) == 3:
        arr_resiz_bw = cv2.cvtColor(arr_resiz, cv2.COLOR_BGR2GRAY)
    else:
        arr_resiz_bw = arr_resiz

#http://answers.opencv.org/question/27411/use-opencv-to-detect-text-blocks-send-to-tesseract-ios/
    #clear the pic
    #arr_resiz_bw = morphOp(arr_resiz_bw)
    bw_denoise = cv2.fastNlMeansDenoising(arr_resiz_bw)
    printPic(bw_denoise ,stepSet,'bw_denoise')

##    bw = cv2.adaptiveThreshold(bw_denoise,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
##    printPic(bw ,stepSet,'bw_contrast_255')
    bw = cv2.adaptiveThreshold(bw_denoise,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                               ,cv2.THRESH_BINARY,11,2)
    printPic(bw ,stepSet,'bw_contrast_255')
    
    arr_resiz_bw = erodeOp(arr_resiz_bw,50)
    arr_resiz_bw = blurOp(arr_resiz_bw,50)
    #bw_blur = blurOp(bw,2)
    bw_blur = morphOp(bw,5)
    
##    bw = cv2.adaptiveThreshold(bw_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
##    printPic(bw ,stepSet,'bw_contrast_1_threshold')
    #cut out black border
    
    #bw_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img, contours,hierarchy = cv2.findContours(bw,cv2.RETR_EXTERNAL
                                               ,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    cv2.imwrite('sofwinres.png',crop)

    arrAlign = rotateOp(bw)

    #Processing starts here:
    #morp= morphOp(arr_resiz)
    bw_Hough,lines= cannyHoughOp(arrAlign)
    print("we found "+str(len(lines)) +' lines')
    #mask_erode = morphOp2(bw_Hough,5)
    #Find external contours from the mask, which most probably will belong to tables or to images
    
    bwOrgSiz = downscale_image(bw_Hough, H=orgShape[0],W=orgShape[1])
    cv2.imwrite(timestampstr+'bwOrgSiz.jpg',bwOrgSiz)
    cv2.imshow("bw_orgsize", bwOrgSiz)
    stepSet.append("resize")
    cv2.waitKey(0)

    with open(timestampstr+'output.txt','w') as f:
        f.write(str(stepSet))
#http://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document

def main():
    #http://stackoverflow.com/questions/23506105/extracting-text-opencv
    #http://answers.opencv.org/question/27411/use-opencv-to-detect-text-blocks-send-to-tesseract-ios/
    #path = '2004FLG-1.png'
    #path = './BL/Bill of Lading2Bill-of-Lading-Origin.jpg'
    path = './BL/Bill of Lading3Bill-of-lading-full.jpg'
    kernel = np.zeros((2,2), np.uint8)
    blurIter = 5
    #stepSet=[]
    os.makedirs(timestampstr)

    arr = cv2.imread(path,0) #read in gray color
    orgShape = arr.shape
    print(orgShape)
    
    printPic(arr,stepSet,'org')
    
    #resize image
    arr_resiz = downscale_image(arr, H=660,W=510)
    printPic(arr_resiz,stepSet,'resiz')

    #change to mono color
    if len(arr_resiz.shape) == 3:
        arr_resiz_bw = cv2.cvtColor(arr_resiz, cv2.COLOR_BGR2GRAY)
    else:
        arr_resiz_bw = arr_resiz

    #remove the gray color noise
    arr_thre = thresholdOp(arr_resiz,2)

    #Morph / remove the small dots
    arr_mob = morphOp(arr_thre,"W")
    #Strengthen the lines
    arr_sob = sobelOp(arr_mob) #black background
    #Blur the images
    arr_er = blurOp(arr_sob,10,"W")
    #Spot the lines (Canny), Hough (project the lines) is not used
    arr_H,lines = cannyHoughOp(arr_er)

    #my take is that canny processing is enough for OCR Project.\
    #we only need to check if there is any blocking white dots in between
    
    #Spot the objects
    arr_cont,contours = contoursOp(arr_H)
    outbw = cv2.drawContours(arr_resiz,contours, -1, (0,255,0), 3)
    printPic(outbw,stepSet,'outbw_contours')

    #combine the contours to be a larger blocks
    #arr_blurred = cv2.GaussianBlur(arr_cont, (5, 5), 0)
    bwOrgSiz = downscale_image(outbw, H=orgShape[0],W=orgShape[1])
    printPic(bwOrgSiz,stepSet,'Org_Size')

    with open(timestampstr+'contours.txt','w') as f:
        f.write(str(arr_cont))
    with open(timestampstr+'steps.txt','w') as f:
        f.write(str(stepSet))
    print('finished')

        
main()
#http://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
