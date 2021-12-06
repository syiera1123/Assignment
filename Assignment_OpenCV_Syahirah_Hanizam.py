import cv2
import numpy as np

count = 0

#-- No.1 Load image --#
img_ori = cv2.imread('OpenCV_Assignment_Image.png')
grey = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
grey1 = cv2.GaussianBlur(grey, (7, 7), 0)

font = cv2.FONT_HERSHEY_COMPLEX

dimensions = grey.shape
height = grey.shape[0]
width = grey.shape[1]

x1 = 325
y1 = 67
x2 = 660
y2 = 324

x3 = 131
y3 = 37
x4 = 600
y4 = 339
    
p1 = (x1, y1)
p2 = (x2, y2)
p3 = (x3, y3)
p4 = (x4, y4)

cropped_img1 = grey1[y1:y2, x1:x2]
cropped_img2 = grey1[y3:y4, x3:x4]

edges = cv2.Canny(cropped_img1, 50, 100)
edges1 = cv2.Canny(cropped_img2, 50, 100)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(edges, kernel)
mask = cv2.erode(mask, kernel)

mask1 = cv2.dilate(edges1, kernel)
mask1 = cv2.erode(mask1, kernel)

#-- No.2 Detect circle (Wood log) --#
detected_circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT, 1, minDist=15, param1=50, param2=20, minRadius=10, maxRadius=38)

if detected_circles is not None:
  
    detected_circles = np.uint16(np.around(detected_circles))
  
    for pt in detected_circles[0, :]:
        count += 1
        
        a, b, r = pt[0], pt[1], pt[2]
        
        #-- highlight using green rectangle --#
        cv2.rectangle(img_ori, (a+x1-r, b+y1-r), (a+x1+r, b+y1+r), (0, 255, 0), 2)
        
        cv2.imshow('Detected Circle', img_ori)
                
        key=cv2.waitKey(1)
        if key%256 == 27:
            break

#-- No.3 Detect rectangle --#
_, contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if area > 400:
        
        if len(approx) == 4:
            #-- highlight using blue rectangle --#
            cv2.drawContours(img_ori, [approx], 0, (255, 0, 0), 2)
            cv2.putText(img_ori, 'Rectangle', (x, y), font, 1, (0, 0, 0))
            

# cv2.imshow('canny', edges)
cv2.imshow('Frame', img_ori)
print('Detected Wood Log = ', count)

cv2.waitKey(0)
cv2.destroyAllWindows()
