#include <iostream>
#include <opencv2/opencv.hpp>

#using namespace std;
import cv2
import numpy as np
import datetime
import os

timestampstr = "{:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
timestampstr=timestampstr+'/'

stepSet=[]
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
    #canny + hough
    bw_Canny = cv2.Canny(bw_blurred,50,150,apertureSize = 3)
    cv2.imwrite(timestampstr+'bw_Canny.jpg',bw_Canny)
    cv2.imshow("bw_Canny", bw_Canny)
    stepSet.append("Canny")
    cv2.waitKey(0)

    bw_Hough = bw_Canny #keep a copy to avoid Canny being updated...after hough
    minLineLength = 100
    maxLineGap = 10
    #retval, bestLabels, centers = cv2.kmeans(getLineDataset(lines),K=20,criteria = 20,attempts=20,flags=cv2.KMEANS_RANDOM_CENTERS)
    lines = cv2.HoughLinesP(bw_Hough,1,np.pi/180,100,minLineLength,maxLineGap)
    
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(bw_Hough,(x1,y1),(x2,y2),150,2)
        
    #lines = cv2.HoughLines(bw_Hough,1,np.pi/180,200)

    #lines = cv2.HoughLinesP(lines,1,0,100,minLineLength,maxLineGap)
    #bw_hough= cv2.HoughLines(lines,1,0,200)
    #bw_Canny = cv2.erode(bw_Canny, kernel = kernel, iterations=1)

    #find the longest horizontal line, rotate by affine transformation to horiztional panel
    #lines = sorted(lines, key = cv2.contourArea, reverse = True)[:5]
    
    cv2.imwrite(timestampstr+'bw_Hough.jpg',bw_Hough)
    cv2.imshow("bw_Hough", bw_Hough)
    stepSet.append("Hough")
    cv2.waitKey(0)
    return bw_Hough, lines

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

def morphOp(bw):
    ##Apply morphology operations
    kernel = np.ones((2,2), np.uint8)
    blurIter= 5
    bw_blurred = cv2.dilate(bw, kernel = kernel, iterations=blurIter*2)
    bw_blurred = cv2.erode(bw_blurred, kernel = kernel, iterations=blurIter*1)
    #bw_blurred = cv2.GaussianBlur(bw_blurred, (5, 5), 0)
    bw_blurred = cv2.erode(bw_blurred, kernel = kernel, iterations=blurIter*1)
    cv2.imwrite(timestampstr+'bw_blurred.jpg',bw_blurred)
    cv2.imshow("bw_blurred", bw_blurred)
    stepSet.append("dilate + erode")
    cv2.waitKey(0)
    return bw_blurred

def morphOp2(bw_blurred):
        #bw_blurred = bw
    vertical = bw_blurred
    horizontal = bw_blurred
    scale = 100
    blurIter= 5
    kernel = np.ones((2,2), np.uint8)

#    for scale in range(1,501,10):
    horizontalsize = int(round(horizontal.shape[1]/scale))
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=blurIter*2)
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations=blurIter*2)
    
    ##Show extracted horizontal lines
    cv2.imwrite(timestampstr+'horizontal.jpg',horizontal)
    cv2.imshow("horizontal"+str(scale), horizontal)
    stepSet.append("dilate + erode - horizontal")
    cv2.waitKey(0)
    
    
    useVertical = True
    if useVertical == True:
        #comment out - no need to apply vertical
        scale = 500
    #        scale +=5
        verticalsize = int(round(vertical.shape[1]/scale))
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize))
        vertical = cv2.dilate(vertical, verticalStructure, iterations=blurIter*2)
        vertical = cv2.erode(vertical, verticalStructure, iterations=blurIter*2)

        
        ##Show extracted vertical lines
        cv2.imwrite(timestampstr+'vertical.jpg',vertical)
        cv2.imshow("vertical"+str(scale), vertical)
        stepSet.append("dilate + erode - vertical")
        cv2.waitKey(0)

        #mask
        mask = vertical + horizontal
        ##Show extracted vertical lines


        mask_erode = cv2.erode(mask, kernel = kernel, iterations=blurIter)
    else:
        mask_erode = horizontal
    cv2.imshow("mask"+str(scale), mask_erode)
    cv2.imwrite(timestampstr+'mask_erode.jpg',mask_erode)
    stepSet.append("dilate + erode - mask2")
    cv2.waitKey(0)    
       
    return mask_erode

def contoursOp(bw_blurred):
    ret, thresh = cv2.threshold(bw_blurred, 127, 255, 0)
    bw_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(str(len(contours)) +' contours found!')

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:]
    for cnt in contours:
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,closed = True)
        bw_cont = cv2.drawContours(bw_img,[approx],0,(150),2)

    cv2.drawContours(bw_img,contours, -1, (0,255,0), 3)
    cv2.imshow("bw_img.jpg", bw_img)
    cv2.imwrite(timestampstr+'bw_img.jpg',bw_img)
    stepSet.append("Contours")
    cv2.waitKey(0)
    return bw_img
    
def main():
    #path = '2004FLG-1.png'
    #path = './BL/Bill of Lading2Bill-of-Lading-Origin.jpg'
    path = './BL/Bill of Lading3Bill-of-lading-full.jpg'
    kernel = np.ones((2,2), np.uint8)

    os.makedirs(timestampstr)

    arr = cv2.imread(path)
    orgShape = arr.shape
    print(orgShape)
    cv2.imwrite('org.jpg',arr)
    
    arr_resiz = downscale_image(arr, H=660,W=510)
    cv2.imwrite(timestampstr+'bw_resiz.jpg',arr_resiz)
    cv2.imshow("bw_resiz", arr_resiz)
    stepSet.append("resize")
    cv2.waitKey(0)

    if len(arr_resiz.shape) == 3:
        arr_resiz_bw = cv2.cvtColor(arr_resiz, cv2.COLOR_BGR2GRAY)
    else:
        arr_resiz_bw = arr_resiz


    #Processing starts here:
    #morp= morphOp(arr_resiz)
    bw_Hough,lines= cannyHoughOp(arr_resiz)
    print("we found "+str(len(lines)) +' lines')
    mask_erode = morphOp(bw_Hough)
    #Find external contours from the mask, which most probably will belong to tables or to images
    #mask_erode = morphOp2(bw_Hough)
    
    #bw_Contours = contoursOp(mask_erode)
    #bw_Contours = connectnearContours(mask_erode)
    
    bwOrgSiz = downscale_image(mask_erode, H=orgShape[0],W=orgShape[1])
    cv2.imwrite(timestampstr+'bwOrgSiz.jpg',bwOrgSiz)
    cv2.imshow("bw_orgsize", bwOrgSiz)
    stepSet.append("resize")
    cv2.waitKey(0)

    with open(timestampstr+'output.txt','w') as f:
        f.write(str(stepSet))
        
main()
#http://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
