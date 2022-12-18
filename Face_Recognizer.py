import cv 2
import os
import pickle
from datetime import datetime
now=datetime.now()
face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")

labels={"Name":1}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1:

    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    def time():
        c_time = now.strftime("%H:%M:%S")
        return c_time
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color= img[y:y+h, x:x+w]
        id_,conf=recognizer.predict(roi_gray)
        if conf>45 and conf<=90 :
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            #c_time=time()
            cv2.putText(img,time(),(50,50),font,1,(0,0,255))
            cv2.putText(img,name,(x,y),font,1,color)
        img_item="my-image.png"
        cv2.imwrite(img_item,roi_color)


    cv2.imshow('img',img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

cv2.destroyAllWindows()