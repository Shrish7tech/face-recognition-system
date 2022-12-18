
import cv2
import pickle
import os
from PIL import Image
import numpy as np

cur_id=0
label_ids={}
y_labels=[]
x_train =[]
# file name
base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"images")

face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
for root, dirs,files in os.walk(image_dir):
      for file in files:
          if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
              path=os.path.join(root,file)
              label=os.path.basename(root).replace("_"," ")

              print(label)


              if not label in label_ids :
                  label_ids[label]=cur_id
                  cur_id+=1
              id_=label_ids[label]
              #print(label_ids)

              pil_image= Image.open(path).convert("L") #converting images from data set into grayscale image
              image_array=np.array(pil_image,"uint8")
              #print(image_array)
              faces = face_cascade.detectMultiScale(image_array, 1.2, 4)
              for (x,y,w,h) in faces:
                     roi=image_array[y: y+h, x:x+w]
                     x_train.append(roi)
                     y_labels.append(id_)

print(y_labels)
#print(x_train)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("train.yml")





