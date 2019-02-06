import cv2
import numpy as np
from keras.models import model_from_json
import json
from face_recog import *

database = {}
data = []
#Loading the model and weights
json_file = open("face.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("face.h5")

try:
    with open('data.txt') as json_file:  
        data = json.load(json_file)
    for name in data:
        path = 'known_faces/' + name + '.jpg'
        database[name] = img_path_to_encoding(path, model)
except:
    print("database empty")

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while (True):
    ret, frame = cap.read()
    if (True):
            #Getting locations of each face in the frame
        #frame = frame[:, :, ::-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        #Getting encoding for each face in the frame
        for (x, y, w, h) in faces:
            face = frame[y:(y+h), x:(x+w)]
            _, name, color = who_is_it(face, database, model)
            cv2.imshow(name, face)
            display(frame, x, y, w, h, name, color)
            if name == "Unknown":
                c = cv2.waitKey(0)
                print("Found a face at", x,y,x+w,y+h, "Press 's' to save", sep = ' ')
                if c & 0xFF == ord('s'):
                    name = input('Name:')
                    data.append(name)
                    database[name] = img_path_to_encoding(path, model)
                    path = 'known_faces/'+ name + '.jpg'
                    cv2.imwrite(path, face)
    cv2.imshow("Video",frame)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break
cap.release()


with open('data.txt', 'w') as outfile:  
    json.dump(data, outfile)
    
cv2.destroyAllWindows()