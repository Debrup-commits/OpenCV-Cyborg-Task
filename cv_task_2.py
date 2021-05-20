import cv2
import numpy as np
import time

cap = cv2.VideoCapture('vid1.mp4')

if cap.isOpened() == False:
    print("Error in opening video...")

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)

        #creating a black image with dimensions same as the loaded input image
        black_img = np.zeros((frame.shape[0], frame.shape[1], 3))

        #creating kernel
        kernel = np.ones((30, 30), np.uint8)

        #grayscale conversion
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #morphing
        closing = cv2.morphologyEx(grayscale_frame, cv2.MORPH_CLOSE, kernel)
        
        #binarization of the grayscale morphed image
        ret, threshold = cv2.threshold(closing, 128, 255, cv2.THRESH_BINARY_INV)
        
        #canny edge detection
        canny_edg_frame = cv2.Canny(threshold, 30, 180)
        
        #identifying contours
        contours, hierarchy = cv2.findContours(canny_edg_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #Drawing contours over black image
        cv2.drawContours(black_img, contours, -1, (0, 0, 0), 3)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 100:
                M = cv2.moments(cnt)
                
                if M['m00'] != 0:
                    centroid_x = int(M['m10']/M['m00'])
                    centroid_y = int(M['m01']/M['m00'])
                
                    cv2.circle(black_img, (centroid_x, centroid_y), 5, (255, 255, 255), -1)
                    cv2.putText(black_img, "("+str(centroid_x)+","+str(centroid_y)+")", (centroid_x+5, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    
                cv2.imshow('black_vid', black_img)
                


        
        if cv2.waitKey(25) and 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
