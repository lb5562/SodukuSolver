from utils import *
import cv2 as cv
import numpy as np

pathImage = "Puzzles/Sudoku5.jpg"
pathImage2 = "Puzzles/SudMedium1.jpg"
pathImage3 = "Puzzles/Sudoku3.jpg"
pathImage4 = "Puzzles/Sudoku21.jpg"

heightImg = 450
widthImg = 450
model = intializePredictionModel()

img = cv.imread(pathImage)
img = cv.resize(img, (widthImg, heightImg))


def formImage(pathImg):
    image = cv.imread(pathImg)
    image = cv.resize(image, (widthImg, heightImg))
    return image


img2 = formImage(pathImage2)
img3 = formImage(pathImage3)
img4 = formImage(pathImage4)
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Debugging
imgThres = preProcess(img)
# Finding Corners
# def findCorners(image):
imgContours = img.copy()
imgBigCon = img.copy()
contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 233, 0), 3)

# Find Biggest Box
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    cv2.drawContours(imgBigCon, biggest, -1, (0, 255, 0), 10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpC = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDigits = imgBlank.copy()
    imgWarpC = cv2.cvtColor(imgWarpC, cv2.COLOR_BGR2GRAY)

##Fingind Digits
imgSolved = imgBlank.copy()
boxes= splitBoxes(imgWarpC)
print(len(boxes))
numbers = getPreduction(boxes,model)
#imgDetectedDigits = displayNumbers(imgDigits,numbers,color=(255,0,255))
#numbers = np.asarray(numbers)
#posArray = np.where(numbers>0,0,1)


imageArray = ([img, imgThres, imgContours, imgBigCon],
              [imgWarpC, imgBlank, imgBlank, imgBlank])
stackedImages = stackImages(1, imageArray)
cv2.imshow('Stacked Images', stackedImages)
cv2.waitKey(0)
