import cv2
import os
import numpy as np
import pickle


cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
casc2Path = cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(casc2Path)
casc3Path = cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_smile.xml"
smile_cascade= cv2.CascadeClassifier(casc3Path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
print("here")
#faceCascade = cv2.CascadeClassifier(cascPath)
labels = {}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frames = cap.read()
    #print("here2")
    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x)
        roi_gray = gray[y:y+h,x:x+w] #(ycoord_start, ycoord end)
        roi_color = frames[y:y+h,x:x+w]# pixels take the actual square from the x and the y axis 

        # we use a deep learning model here to recognize faces 
        id_, conf = recognizer.predict(roi_gray) # predict a region of intrest
        if conf >=45:

            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color=(255,255,255)
            stroke = 2
            cv2.putText(frames,name,(x,y),font,1,color,stroke,cv2.LINE_AA)


       # print("something here ")
        
           
         
        
        
        img_item = "my-image.heic"
       # cv2.imwrite(img_item,roi_gray)
        color =(255,0,0) #BGR 0-255
        stroke = 2
        end_cord_x = x + w # width
        end_cord_y = y + h  # height
        cv2.rectangle(frames,(x,y),(end_cord_x,end_cord_y),color,stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        
        

    

   
    # Display the resulting frame
    cv2.imshow('frames', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break