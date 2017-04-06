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
import math
from scipy.ndimage.filters import rank_filter
import subprocess #for calling C program


global stepSet
stepSet = []

timestampstr = "{:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
timestampstr=timestampstr+'/'
fileName = 'shipping_seflBillofLading.pdf'
timestampstr = '/home/bnpp/ReadingForms/01_ObservatoryData/Blocks_Pred/2017-04-06_17:23:50/'

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

        refLn = [0,0,0,0,0] #x1,y1,x2,y2,angle --> this line is used for rotation
        maxLn = 0
        for line in lines:
            x1,y1,x2,y2 = line[0]
            print(line[0])
            cv2.line(bw_Hough,(x1,y1),(x2,y2),255,2)
            lengthLine = math.sqrt(math.pow((y2-y1),2)+math.pow((x2-x1),2))
            angle = math.atan2((y2-y1),(x2-x1))
            print('line: ' + str(refLn) + 'Angle '+str(angle))
            if lengthLine > maxLn and math.fabs(angle)<0.785: #line being long and angle is only 45 degrees
                maxLn = lengthLine
                refLn[0]=x1
                refLn[1]=y1
                refLn[2]=x2
                refLn[3]=y2
                refLn[4]= angle
                print('max line ' + str(refLn))
            
        bw_out = bw_Hough
        printPic(bw_Canny,stepSet,'bw_Hough_Canny')
    else:
        bw_out = bw_Canny
    return bw_out, lines


def blurOp(bw,blurIter = 1,fontColor = "W",struct=False):
    ##remove noises
    ##Apply morphology operations
    if struct == True:
       kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,3)) 
    elif fontColor == "W":
        kernel = np.ones((2,2), np.uint8)
    else:
        kernel = np.zeros((2,2), np.uint8)    
   
    bw_blurred = cv2.dilate(bw, kernel = kernel, iterations=blurIter,anchor=(0,0))

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

    bw_blurred = cv2.erode(bw, kernel = kernel, iterations=blurIter,anchor=(0,0))

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
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations=blurIter*2,anchor=(0,0))
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=blurIter*2,anchor=(0,0))
    
    ##Show extracted horizontal lines
    printPic(horizontal,stepSet,"horizontal")
    
    useVertical = True
    if useVertical == True:
        scale = 50

        #scale +=5
        verticalsize = int(round(vertical.shape[1]/scale))
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize))
        vertical = cv2.erode(vertical, verticalStructure, iterations=blurIter*2,anchor=(0,0))
        vertical = cv2.dilate(vertical, verticalStructure, iterations=blurIter*2,anchor=(0,0))
        
        ##Show extracted vertical lines
        printPic(vertical,stepSet,"vertical")

        #mask
        mask = vertical + horizontal
        ##Show extracted vertical lines


        mask_erode = cv2.erode(mask, kernel = kernel, iterations=blurIter,anchor=(0,0))
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

def rotateOp(arr,lines):
    
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
def rotateOp2(arr, contours):
    
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        ret = cv2.drawContours(arr,[box],0,150,2)
        #ret = drawn contours
        rectC =rect[0]
        rectSize =rect[1]
        recAng = rect[2]
        print(rect)

    #rotate
    Rotate = False
    if Rotate == True:
        rot_mat = cv2.getRotationMatrix2D(rectC,recAng,1.0)
        ret = cv2.warpAffine(ret, rot_mat, arr.shape,flags=cv2.INTER_LINEAR)
    #rotate the points?
    
    printPic(ret,stepSet,'bwRotate')

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

def find_border_components(contours, ary):
    #remove anything outside of the border
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders

def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)

def dilate(ary, N, iterations): 
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    #I have swapped 1 and 0
    kernel = np.ones((N,N), dtype=np.uint8)
    kernel[int((N-1)/2),:] = 0
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations,anchor=(0,0))
    printPic(dilated_image,stepSet,'dilateImg1')
    
    kernel = np.ones((N,N), dtype=np.uint8)
    kernel[:,int((N-1)/2)] = 0
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations,anchor=(0,0))

    printPic(dilated_image,stepSet,'dilateImg2')
    return dilated_image

def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.

    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.

    count = 21
    dilation = 5
    n = 1
       
    dilated_image = dilate(edges, N=3, iterations=2)
    dilated_image = dilated_image.astype('uint8') ##check--->why is this output so bad...
    tmpOut,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(str(len(contours)) +' found')
    printPic(edges,stepSet,'find_components')

    return contours
def find_components_old(edges, max_components=16):
    """Dilate the image until there are just a few connected components.

    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.

    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        
        dilated_image = dilate(edges, N=3, iterations=n)

        #change to mono color
##        tmpname = 'tmp'+str(randint(0,100))+'.jpg'
##        cv2.imwrite(timestampstr+tmpname,dilated_image)
##        dilated_image= cv2.imread(timestampstr+tmpname,0)
        dilated_image = dilated_image.astype('uint8')
##        tmpname = 'tmp'+str(randint(0,100))+'.jpg'
##        cv2.imwrite(timestampstr+tmpname,dilated_image)
##        dilated_image= cv2.imread(timestampstr+tmpname,0)
        printPic(dilated_image,stepSet,'dilateImgb4contours')
        tmpOut,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        printPic(tmpOut,stepSet,'dilated_img'+str(n)+'_'+str(count))
        count = len(contours)
    printPic(edges,stepSet,'find_components')
    #print dilation
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours

def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))

def extractBlocks(timestampstr, path):
    #path = '2004FLG-1.png'
    #path = './BL/Bill of Lading2Bill-of-Lading-Origin.jpg'
    #path = './BL/Bill of Lading3Bill-of-lading-full.jpg'
    kernel = np.zeros((2,2), np.uint8)
    blurIter = 5
    #stepSet=[]
    os.makedirs(timestampstr)

    arr = cv2.imread(path,0) #read in gray color
    orgShape = arr.shape
    print(orgShape)
    
    printPic(arr,stepSet,'org')
    imgH = 660
    imgW = 510
    
    #resize image
    arr_resiz = downscale_image(arr, H=imgH,W=imgW)
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

    '''**************Testing here**************'''
    #find components
    edges_org = cv2.Canny(np.asarray(arr), 100, 200) #canny will change its black and white
    _,contours, hierarchy = cv2.findContours(edges_org, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, edges_org)
    #change to mono color
##    tmpname = 'tmp'+str(randint(0,100))+'.jpg'
##    cv2.imwrite(timestampstr+tmpname,edges_org)
##    edges= cv2.imread(timestampstr+tmpname,0)
    edges = np.uint8(edges_org)

    #remove anthing outside the border of the contours
    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)
        
   # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered

    #find components
    contours = find_components(edges)
    if len(contours) == 0:
##        print '%s -> (no text!)' % path
        return
    components = cv2.drawContours(arr_resiz,contours, -1, (0,255,0), 3)
    printPic(components,stepSet,'components')   
    '''*******************************'''
    
    #Spot the objects    
    arr_cont,contours = contoursOp(arr_H)
    outbw = cv2.drawContours(arr_resiz,contours, -1, (0,255,0), 3)
    printPic(outbw,stepSet,'outbw_contours')
    #outbw = cv2.drawContours(arr_resiz,contours, -1, (0,255,0), 3)
    img = np.ones(shape = (imgH,imgW),dtype ='uint8')
    img_shape = cv2.drawContours(img,contours, -1, 255, 3)
    img_shape1 = downscale_image(img_shape, H=orgShape[0],W=orgShape[1])
    printPic(img_shape1,stepSet,'img_shape1')

    #combine the contours to be a larger blocks
    #arr_blurred = cv2.GaussianBlur(arr_cont, (5, 5), 0)
    bwOrgSiz = downscale_image(outbw, H=orgShape[0],W=orgShape[1])
    printPic(bwOrgSiz,stepSet,'Org_Size')

    with open(timestampstr+'contours.txt','w') as f:
        f.write(str(arr_cont))
    with open(timestampstr+'steps.txt','w') as f:
        f.write(str(stepSet))
    print('finished')

    return img_shape

#put in list of labels's xy (beginning xy)
# put in a particular word
class projectionArea:
    def calColorChg(self,label, word):
        xw, yw = word
        xl, yl = label
        areaCnt = 0
        for y in range(yl,yw+1):
            for x in range(xl,xw+1):
                print(str(x)+','+str(y))
                if self.img[x][y] > 0:
                   areaCnt +=1 
        return areaCnt

    def __init__(self,img, wordsxy, labelsxy):
        self.labelsxy = labelsxy
        self.wordsxy = wordsxy
        self.img = img
        listlabel = len(labelsxy)
        listword = len(wordsxy)
        self.xysets = np.ones(shape=(listword,listlabel),dtype='uint8')
        #2 d matrix that list the distance between words and labels
        
        for i in range(listword):
            xw, yw = self.wordsxy[i]
            for j in range(listlabel):
                xl, yl = self.labelsxy[j]
                if yl <= yw and xl <= xw: #pick the right pair
                    print('right pair')
                    self.xysets[i][j]= self.calColorChg(self.labelsxy[j],self.wordsxy[i])
                    
    

        

words = [(30,30),(10,10),(20,20)]
labels = [(2,3) ,(4,4)]
imgpth = timestampstr+'bw_Hough_Canny.jpg'
img= cv2.imread(imgpth,0)
PA = projectionArea(img, words, labels)
print(PA.xysets)
#debug
##subprocess.call(["/bin/bash","P01_convertPDFToXML",fileName]) ##OCR the files selected        
##imgblock = extractBlocks(timestampstr=, path = timestampstr+fileName+'.png')
##noColorChg = calColorChg(img, startxy, labelxy)

#http://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
#http://stackoverflow.com/questions/23506105/extracting-text-opencv
#http://answers.opencv.org/question/27411/use-opencv-to-detect-text-blocks-send-to-tesseract-ios/
#http://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
