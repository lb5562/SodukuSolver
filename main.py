from utils import *
import cv2 as cv
import numpy as np

pathImage = "Puzzles/SudMedium1.jpg"
heightImg = 450
widthImg = 450

img = cv.imread(pathImage)
img = cv.resize(img, (widthImg, heightImg))

imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Debugging
imgThres = preProcess(img)

imageArray = ([img,imgBlank,imgBlank,imgBlank],
              [imgBlank,imgBlank,imgBlank,imgBlank])
stackedImages = stackImages(1,imageArray)
cv2.imshow('Stacked Images',stackedImages)
cv2.waitKey(0)
