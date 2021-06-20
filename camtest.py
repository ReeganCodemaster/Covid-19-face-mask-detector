'''
Use this file to detrmine id of video stream for main program
'''

import cv2 
# Open the device at the ID 

#start searching for video stream from 0 upwards
ID = 0
cap = cv2.VideoCapture(ID)

#Check whether user selected camera is opened successfully.

if not (cap.isOpened()):

    print("Could not open video device")



while(True):

# Capture frame-by-frame

    ret, frame = cap.read()

    # Display the resulting frame

    cv2.imshow('preview',frame)

    #Waits for a user input to quit the application

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

# When everything is done, release the capture

cap.release()

cv2.destroyAllWindows()