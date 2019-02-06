import cv2
import numpy as np



cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def display(x, y, w, h, color):
        # Draw a box around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom + 20), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while (True):
    ret, frame = cap.read()
    if (True):
            #Getting locations of each face in the frame
        #frame = frame[:, :, ::-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        #Getting encoding for each face in the frame
        for (x, y, w, h) in faces:
            img = frame[y:(y+h), x:(x+w)]
            cv2.imshow("face", img)
            print(x,y,x+w,y+h, sep = ' ')
            display(x, y, w, h, (255, 255, 255))
        
            #Lock if trusted face is not found 
    cv2.imshow("Video",frame)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()