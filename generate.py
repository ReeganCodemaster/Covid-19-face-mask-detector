import os
import cv2
import numpy as np

# path joining version for other paths
ORIGINAL_DIR = 'C:\\Users\\reega\\OneDrive\\Programing\\Code\\python\\Face Msk dtector\\Traning data\\original_data'
COMPLETED_DIR = 'C:\\Users\\reega\\OneDrive\\Programing\\Code\\python\\Face Msk dtector\\Traning data\\data'

#defining prototext and caffemodel paths
caffeModel = "Face detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
prototextPath = "Face detector/deploy.prototxt.txt"

#loading face detction model
print('loding models............\n')
net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

#loop over folders in the original_data dir
with os.scandir(ORIGINAL_DIR) as folders:
    for folder in folders:
        file_num = 0
        print('Working on ' + folder.name + '..........\n')
        if os.path.isdir(os.path.join(ORIGINAL_DIR, folder)):

            #looping over each folder in original_data dir
            with os.scandir(os.path.join(ORIGINAL_DIR, folder)) as files:
                for file in files:
                    file_num += 1

                    file_name = file.name
                    file_path = os.path.join(ORIGINAL_DIR, folder.name, file_name)
                    img = cv2.imread(file_path)

                    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
                    (h, w) = img.shape[:2]
                    # blobImage convert RGB (104.0, 177.0, 123.0)
                    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                            (300, 300), (104.0, 177.0, 123.0))

                    # passing blob through the network to detect and pridiction
                    net.setInput(blob)
                    detections = net.forward()
                    
                    #for each face in the image
                    for i in range(0, detections.shape[2]):

                        #extract confidence
                        confidence = detections[0, 0, i, 2]

                        # filter detections by confidence greater than the minimum confidence
                        if confidence < 0.9 :
                            continue

                        # Determine the (x, y)-coordinates of the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        
                        #creating a cropped of each face with  a name and path
                        cropped_img = img[startX:endX, startY:endY]
                        cropped_img_name = folder.name+ str(file_num) + 'face' +str(i)+'.jpg'
                        cropped_img_path = os.path.join(COMPLETED_DIR, folder.name, cropped_img_name)

                        #try to save the cropped image to a file
                        try:
                            status = cv2.imwrite(cropped_img_path, cropped_img)
                            if status == True:
                                print('Saved ' + cropped_img_path + '............\n')
                            else:
                                print(cropped_img_path + 'save failed ............\n')
                        except:
                            print('file save error')
        
