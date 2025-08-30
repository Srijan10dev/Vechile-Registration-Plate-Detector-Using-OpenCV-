## A snippet for detecting the vechile regirstration plate and croping it to enchance it's visibility
import cv2
## We will be using the pre-trained haarcasade 'haarcascade_russian_plate_number.xml' cascade for better dectection 
## Avaible in https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml
plate_cascade=cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') ## using a russian cascade for a default cascade model

img=cv2.imread('car3.jpg') ## Loads the Images in the computer vision (Opencv)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ## Converts the image to grayscale for better quatity and accurate detection

plate=plate_cascade.detectMultiScale(gray,1.1,4) ##used for detecting the cascade running the frame

for (x, y, w, h) in plate:  ## Running a loop for drawing the rectangle across the plate and also showing the plate
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) ## Draving a rectangle for better detection

cv2.imshow('Plate', img[y:y+h, x:x+w]) ## Showing the Plate (Cropped) in other dialog box 

## Improving the scaling factor for resizing the image 
max_dimension = 500 ##setting the dimensional area for convention
height, width = img.shape[:2] ## in the array we have height, width , and chanels for core coloring,, we extract height and width
scaling_factor = max_dimension / float(max(height, width)) ## Calculating the scaling factor

if scaling_factor < 1:  ##scaling factor can't be less than 1 , checking whether corrupted
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) ## Resizeing the image for limited vision
## Setting the new img dimension to the new scaling factor in respect to the Y direction and X direction 

cv2.imshow('Image', img) ## Display the inputted image
cv2.waitKey(0)  ## keeps the terminal opening unless any key is input
cv2.destroyAllWindows() ## Ending the process
## End of Statement