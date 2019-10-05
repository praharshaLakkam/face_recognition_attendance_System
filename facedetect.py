import cv2
import numpy as np
faceDetect=cv2.CascadeClassifier('praharsha.xml');
cam=cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read(r"C:\Users\HP\Desktop\New folder\recognizer\trainingdata.yml")
id=0
s={''}
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
           id="Praharsha"
           s.add("Praharsha")
        elif(id==2):
            id="dheeraj"
            s.add("dheeraj")
        elif(id==6):
            id="Sameer Raja"
            s.add("Sameer Raja")
        else:
            id="unknown"
        cv2.putText(img, str(id), (x,y+h), font, 1.0, (255,255,255));
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        print(s)
        break;

cam.release()
cv2.destroyAllWindows()
