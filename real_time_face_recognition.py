import numpy as np
import cv2
import face_recognition
import glob
#Path to the images folder
path = 'faces/*.jpg'
image_filenames = glob.glob(path)
#Getiing the names of the people in the photos from the filename
known_face_names = [i.split('\\')[1].split('.')[0] for i in image_filenames]
#Reading the face images from the folder
known_face_images = [face_recognition.load_image_file(img) for img in image_filenames]
#Getting face encodings for each image
known_face_encodings = [face_recognition.face_encodings(img)[0] for img in known_face_images]
#Used to store information about the faces found in the video feed
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
#Getting permission to access webcam
cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    #Resizing the frame to increase the speed of processing
    small_frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
    #converting from BGR(what opencv uses) to RGB(what face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    #Getting locations of each face in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    #Getting encoding for each face in the frame
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    # Checking if the face is a match for the known face(s)
    for face_encoding in face_encodings:
        faces = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.6)
        name = "Unknown"

        if True in faces:
            first_match_index = faces.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)
    
    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *=2
        right*=2
        bottom*=2
        left*=2
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom + 20), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom + 20), font, 0.75, (255, 255, 255), 1)

    cv2.imshow("Video",frame)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
