#include <iostream>
#include <opencv2/opencv.hpp>

#using namespace std;
import cv2
import numpy as np

def downscale_image(im, H,W):
    """Shrink im until its longest dimension is <= max_dim.

    Returns new_image, scale (where scale <= 1).
    """
    shp = im.shape
    a =shp[0] #Height
    b =shp[1] #Width
##    if max(a, b) <= max_dim:
##        return 1.0, im

    scaleW = 1.0 * W / b
    scaleH = 1.0 * H / a
    scale = min(scaleH,scaleW)
    
##    new_im = np.resize(im,(int(a * scaleH), int(b * scaleW)))
    #new_im = np.resize(im,newShape)

    #opencv takes (Width,Height)
    new_im = cv2.resize(im,(W,H),interpolation = cv2.INTER_AREA)
    return new_im



def main():
    path = '2004FLG-1.png'
    kernel = np.ones((5,5), np.uint8)

    arr = cv2.imread(path)
    print(arr.shape)
##    cv2.imshow("original", arr)
##    cv2.waitKey(0)
    arr_resiz = downscale_image(arr, H=660,W=510)
##    cv2.imshow("rsz", arr_resiz)
##    cv2.waitKey(0)
    if len(arr_resiz.shape) == 3:
        arr_resiz_bw = cv2.cvtColor(arr_resiz, cv2.COLOR_BGR2GRAY)
    else:
        arr_resiz_bw = arr_resiz
##    cv2.imshow("gray", arr_resiz_bw)
##    cv2.waitKey(0)
    bw = cv2.adaptiveThreshold(arr_resiz_bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2);
##    cv2.imshow("bw", bw)
##    cv2.waitKey(0)
    vertical = bw
    

    scale = 10
    for scale in range(1,501,10):
        horizontal = bw
        horizontalsize = int(round(horizontal.shape[1]/scale))
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

        horizontal = cv2.erode(horizontal, horizontalStructure, iterations=10)
        horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=10)
        cv2.imshow("horizontal"+str(scale), horizontal)
        cv2.waitKey(0)
        scale +=5

main()
#http://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
