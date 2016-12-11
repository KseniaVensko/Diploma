import cv2
import numpy as np
import sys
import os

def edge_detect(file_name, tresh_min, tresh_max):
    image = cv2.imread(file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7),0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None,  iterations=1)

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    #for cnt in contours:
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
    
    return w,h,image

if __name__ == '__main__':
	directory = sys.argv[1]
	dimensions = []
	for filename in sorted(os.listdir(directory)):
		w,h,image = edge_detect(directory + '/' + filename, 128, 255)
		dimensions.append([h,w])
		cv2.imshow('res', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	dimensions = np.array(dimensions)
	np.savetxt('hw_objects', dimensions, fmt='%u', delimiter=' ')
