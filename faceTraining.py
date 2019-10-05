import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create();
path='dataset'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for i in imagePaths:
        faceImg=Image.open(i).convert('L')
        faceNP=np.array(faceImg,'uint8')
        ID=int(os.path.split(i)[-1].split('.')[1])
        faces.append(faceNP)
        print(ID)
        IDs.append(ID)
        cv2.imshow("training",faceNP)
        cv2.waitKey(10)
    return IDs,faces

Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
