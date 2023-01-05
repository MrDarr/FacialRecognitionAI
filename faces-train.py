import os 
import cv2
from PIL import Image
import numpy as np
import pickle
# The logic behind this class is to get the images from the files and assign ids to them so we are able to save and recognize these pictures 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Looks for the path of the current file and gets the directory where it is
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
image_dir = os.path.join(BASE_DIR,"images")
# Lists for the files 
recognizer = cv2.face.LBPHFaceRecognizer_create()

y_labels=[] #nums related to labels 
current_id=0
label_ids= {}
x_train=[]  # numbers of pixel values

for root , dirs , files in os.walk(image_dir): # Iterates through files in directory to find the images 
    for file in files:
        if file.endswith("jpeg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()# Get the label of the folder
            #print(label,path)
            if not label in label_ids:


                label_ids[label] = current_id
                current_id+=1

            id_ = label_ids[label]
            #print(label_ids)
            # Converted images into numbers and stored into an array only works with PNG and JPEG rn
            pil_image = Image.open(path).convert("L") # grayscale
            #size = (550,550)
            #final_image=pil_image.resize(size,Image.ANTIALIAS)
            # every image has pixel values and we convert it to gray scale and then we stored them as a list of numbers so we could train those pictures

            image_array = np.array(pil_image,"uint8")
           # print(image_array)
            # detecting faces inside of an image
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for(x,y,w,h) in faces: 
                roi =image_array[y:y+h,x:x+w]
                x_train.append(roi) # now we append to train the values we just need the labels to associate with the training data 
                y_labels.append(id_)
       # print(y_labels)
       # print(x_train)

    
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)



recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")