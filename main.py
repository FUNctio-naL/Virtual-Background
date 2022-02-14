import cv2
from cv2 import resize
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import os

wt, ht = 640, 480
path = "backgrounds/"

# img = cv2.imread("resources/boy.jpg")
# cv2.imshow("output", img)
# cv2.waitKey(0)


imgList = os.listdir(path)
total = len(imgList)
print(imgList)

def getBg(num):
    imgNew = cv2.imread(path + imgList[num%total])
    imgNew = cv2.resize(imgNew, (wt, ht))
    return imgNew

a =0
# im1 = cv2.imread("resources/BGS/img1.jpg")
# cv2.imshow("showImg", im1)
# cv2.waitKey(0)
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 1)
cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation(model=1)
# Set Fps reader on the image
fpsReader = cvzone.FPS()
# bg1 = cv2.imread(path) 
# bg1 = cv2.resize(bg1, (640, 480))
num = total*100
while(1):
    flag, img = cap.read()
    # cv2.imshow("BG",  bg1)
    
    imgOut = segmentor.removeBG(img, getBg(num), threshold= 0.55)

    # # imgRes= np.hstack((img, imgOut))
    imgRes = cvzone.stackImages([img, imgOut], 2, 1)
    fps, imgRes = fpsReader.update(imgRes, color = (0, 0, 255))
    # cv2.imshow("V", img)
    # cv2.imshow("OUt", imgOut)
    cv2.imshow("Result", imgRes)
    key=cv2.waitKey(24)         
    if key == ord('a'):
            num -=1
    if key == ord('d'):
            num +=1
    if key%256 == 27:
            break